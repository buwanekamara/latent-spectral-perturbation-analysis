import os
import torch
from torch.utils.data import Dataset

class LatentTensorDataset(Dataset):
    def __init__(self, processed_real_dir, processed_fake_dir, is_train=True):
        self.file_paths = []
        self.labels = []
        self.is_train = is_train
        
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

    def __getitem__(self, idx):
        # Load the pre-computed [4, 2048] tensor straight into memory
        features = torch.load(self.file_paths[idx], weights_only=True)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        # Zero-Cost Feature Regularization during training
        if self.is_train:
            # Add micro-scale gaussian noise so the MLP doesn't overfit static numbers
            noise = torch.randn_like(features) * 0.01
            features = features + noise

        return features, label