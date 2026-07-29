import os
import torch
from torch.utils.data import Dataset

class LatentTensorDataset(Dataset):
    def __init__(self, processed_real_dir, processed_fake_dir, is_train=True,
                 normalize=False, delta_only=False):
        self.file_paths = []
        self.labels = []
        self.is_train = is_train
        self.normalize = normalize
        # delta_only: drop the orig_cls (content) half, keep only the Delta
        # signal -> feature becomes [4, 1024]. Tests FIRE's content-bias claim.
        self.delta_only = delta_only
        
        for file_name in os.listdir(processed_real_dir):
            if file_name.endswith('.pt'):
                self.file_paths.append(os.path.join(processed_real_dir, file_name))
                self.labels.append(1.0)
                
        for file_name in os.listdir(processed_fake_dir):
            if file_name.endswith('.pt'):
                self.file_paths.append(os.path.join(processed_fake_dir, file_name))
                self.labels.append(0.0)

    def __len__(self):
        return len(self.file_paths)

    @staticmethod
    def normalize_features(features):
        """
        Relative Spectral Resilience normalization.

        The cached tensor is [4, 2048] = concat(orig_cls[1024], delta[1024]) per layer.
        Raw delta magnitude depends on dataset quality (resolution/compression),
        which shifts scores globally on unseen high-quality data. Dividing both
        halves by the per-layer norm of orig_cls converts the feature into
        'fraction of structure lost under spectral stress' - a scale-invariant
        ratio that is comparable across domains.
        """
        half = features.shape[-1] // 2
        orig_cls = features[..., :half]
        delta = features[..., half:]

        # Per-layer norm of the original CLS token: [4, 1]
        orig_norm = orig_cls.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        orig_cls = orig_cls / orig_norm          # unit-norm context
        delta = delta / orig_norm                # relative degradation ratio

        return torch.cat([orig_cls, delta], dim=-1)

    def __getitem__(self, idx):
        # Load the pre-computed [4, 2048] tensor straight into memory
        features = torch.load(self.file_paths[idx], weights_only=True)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        # Normalize BEFORE slicing: the Delta half is divided by the orig_cls
        # norm, so the content half must still be present at this point.
        if self.normalize:
            features = self.normalize_features(features)

        if self.delta_only:
            half = features.shape[-1] // 2
            features = features[..., half:]          # [4, 1024] Delta only

        # Zero-Cost Feature Regularization during training
        if self.is_train:
            if self.normalize:
                # Normalization shrinks the feature scale, so scale noise to match
                noise = torch.randn_like(features) * 0.01 * features.std().clamp_min(1e-8)
            else:
                # Original behaviour: absolute micro-scale gaussian noise
                noise = torch.randn_like(features) * 0.01
            features = features + noise

        return features, label