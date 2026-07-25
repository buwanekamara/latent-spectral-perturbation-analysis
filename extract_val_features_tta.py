"""
Multi-crop TTA feature extraction for validation (fast version).

Speedups vs v1:
  - N_CROPS default 10 -> 5 (score averaging saturates around 5 crops)
  - optional fp16 for the ViT via --fp16 (the FFT stressor stays fp32: its
    magnitudes overflow half precision and produce NaN features)
  - Several images batched per ViT forward pass (GPU stays saturated)
  - DataLoader workers decode images on CPU in parallel with GPU compute

Saves  <class>/img<idx>_crop<c>.pt ; evaluate.py averages crop scores per image.
"""
import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image, ImageFile
from torchvision import transforms

from models.espcn import FrequencyMaskingModule, load_pretrained_fmre
from models.vit_extractor import FrozenViTExtractor

Image.MAX_IMAGE_PIXELS = 300_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = True

CROP = transforms.Compose([
    transforms.RandomCrop(224, pad_if_needed=True, padding_mode='reflect'),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class TTACropDataset(Dataset):
    """Decodes each image once and returns all its crops stacked."""

    def __init__(self, src_dir, n_crops):
        self.files = sorted(f for f in os.listdir(src_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')))
        self.src_dir = src_dir
        self.n_crops = n_crops

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            image = Image.open(os.path.join(self.src_dir, self.files[idx])).convert("RGB")
            crops = torch.stack([CROP(image) for _ in range(self.n_crops)])
        except Exception as e:
            print(f"Unreadable file {self.files[idx]} ({e}); using zeros.")
            crops = torch.zeros(self.n_crops, 3, 224, 224)
        return crops, idx


def run_tta_extraction():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops", type=int, default=5, help="crops per image")
    parser.add_argument("--images-per-batch", type=int, default=4,
                        help="images per ViT forward pass (crops x this = GPU batch)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fp16", action="store_true",
                        help="fp16 for the ViT only. The FFT stressor ALWAYS runs in "
                             "fp32 (its magnitudes overflow half precision -> NaN).")
    parser.add_argument("--real-dir", default="data/val/real")
    parser.add_argument("--fake-dir", default="data/val/fake")
    parser.add_argument("--save-dir", default="processed_data/val_tta")
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TARGET_LAYERS = [11, 13, 15, 17]
    FIRE_WEIGHTS_PATH = "imagenet_w_adm.pt"
    use_amp = args.fp16 and DEVICE.type == "cuda"

    os.makedirs(f"{args.save_dir}/real", exist_ok=True)
    os.makedirs(f"{args.save_dir}/fake", exist_ok=True)

    fmre = FrequencyMaskingModule().to(DEVICE).eval()
    fmre = load_pretrained_fmre(fmre, FIRE_WEIGHTS_PATH, DEVICE)
    vit_extractor = FrozenViTExtractor(TARGET_LAYERS).to(DEVICE).eval()

    for class_folder, src_dir in [("real", args.real_dir), ("fake", args.fake_dir)]:
        dataset = TTACropDataset(src_dir, args.crops)
        loader = DataLoader(dataset, batch_size=args.images_per_batch, shuffle=False,
                            num_workers=args.workers, pin_memory=use_amp)
        print(f"--- {class_folder}: {len(dataset)} images x {args.crops} crops "
              f"(GPU batch {args.images_per_batch * args.crops}, fp16={use_amp}) ---")

        with torch.no_grad():
            for crops_batch, indices in tqdm(loader, desc=class_folder):
                b, n, c, h, w = crops_batch.shape
                flat = crops_batch.reshape(b * n, c, h, w).to(DEVICE, non_blocking=True)

                # FFT stressor in fp32 ONLY: it computes fft(image * 255) whose
                # magnitudes (~1e7) overflow fp16 -> inf, and log(x + 1e-7) where
                # 1e-7 underflows to 0 -> -inf. Both produce NaN features.
                with torch.autocast(device_type=DEVICE.type, enabled=False):
                    pseudo = fmre(flat.float())

                # ViT is safe in fp16 (LayerNorm-based, well-conditioned)
                with torch.autocast(device_type=DEVICE.type, enabled=use_amp):
                    features = vit_extractor(flat, pseudo)      # [b*n, 4, 2048]

                features = features.float()
                if not torch.isfinite(features).all():
                    bad = (~torch.isfinite(features)).any(dim=(1, 2)).nonzero().flatten()
                    raise RuntimeError(
                        f"Non-finite features for {len(bad)} crop(s) in this batch "
                        f"(images {indices.tolist()}). Re-run without --fp16.")

                features = features.reshape(b, n, 4, -1).cpu()
                for i in range(b):
                    img_idx = int(indices[i])
                    for cr in range(n):
                        torch.save(features[i, cr].clone(),
                                   f"{args.save_dir}/{class_folder}/img{img_idx}_crop{cr}.pt")

    print(f"Done. Evaluate with:\n  python evaluate.py "
          f"--real-dir {args.save_dir}/real --fake-dir {args.save_dir}/fake")


if __name__ == "__main__":
    run_tta_extraction()
