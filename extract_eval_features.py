"""
Feature extraction for the model-wise evaluation sets.

Walks:  Evaluation data set/<dataset>/<generator>/{real,fake}/*.png
Caches: processed_eval/<dataset>/<generator>/{real,fake}/feature_N.pt   ([4,2048])

Run this ONCE on your machine (needs the GPU + the frozen ViT). It is
idempotent: a generator already cached is skipped, so you can re-run safely.
Then run evaluate_report.py to produce the detailed report.
"""
import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.espcn import FrequencyMaskingModule, load_pretrained_fmre
from models.vit_extractor import FrozenViTExtractor
from utils.dataset import DeepfakeDataset

# ==================== CONFIG ====================
EVAL_ROOT   = "Evaluation data set"     # input root (your folders)
OUT_ROOT    = "processed_eval"          # cached-feature root (created here)
FIRE_WEIGHTS = "imagenet_w_adm.pt"
TARGET_LAYERS = [11, 13, 15, 17]
# ===============================================


def has_images(d):
    return os.path.isdir(d) and any(
        f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        for f in os.listdir(d)
    )


def discover_subsets(root):
    """Return list of (dataset, generator, real_dir, fake_dir)."""
    subsets = []
    for dataset in sorted(os.listdir(root)):
        dpath = os.path.join(root, dataset)
        if not os.path.isdir(dpath):
            continue
        for generator in sorted(os.listdir(dpath)):
            gpath = os.path.join(dpath, generator)
            real_dir, fake_dir = os.path.join(gpath, "real"), os.path.join(gpath, "fake")
            if has_images(real_dir) and has_images(fake_dir):
                subsets.append((dataset, generator, real_dir, fake_dir))
    return subsets


def extract_subset(dataset, generator, real_dir, fake_dir, models, device):
    fmre, vit = models
    out_real = os.path.join(OUT_ROOT, dataset, generator, "real")
    out_fake = os.path.join(OUT_ROOT, dataset, generator, "fake")

    # Idempotent skip: if both output dirs already have files, move on.
    if has_images_pt(out_real) and has_images_pt(out_fake):
        print(f"  [skip] {dataset}/{generator} already cached")
        return

    os.makedirs(out_real, exist_ok=True)
    os.makedirs(out_fake, exist_ok=True)

    dataset_obj = DeepfakeDataset(real_dir, fake_dir, is_train=False)
    loader = DataLoader(dataset_obj, batch_size=1, shuffle=False, num_workers=2)

    with torch.no_grad():
        for idx, (image, label) in enumerate(tqdm(loader, desc=f"{dataset}/{generator}")):
            image = image.to(device)
            # FFT stressor in fp32 (half precision overflows -> NaN)
            pseudo = fmre(image.float())
            feats = vit(image, pseudo).squeeze(0).cpu()   # [4, 2048]
            folder = out_real if label.item() == 1.0 else out_fake
            torch.save(feats, os.path.join(folder, f"feature_{idx}.pt"))


def has_images_pt(d):
    return os.path.isdir(d) and any(f.endswith(".pt") for f in os.listdir(d))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", default=EVAL_ROOT)
    parser.add_argument("--out-root", default=OUT_ROOT)
    args = parser.parse_args()
    globals()["OUT_ROOT"] = args.out_root

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subsets = discover_subsets(args.eval_root)
    if not subsets:
        print(f"No <dataset>/<generator>/{{real,fake}} folders found under '{args.eval_root}'.")
        return

    print(f"Found {len(subsets)} generator subsets to extract:")
    for d, g, _, _ in subsets:
        print(f"  - {d}/{g}")

    fmre = FrequencyMaskingModule().to(device).eval()
    fmre = load_pretrained_fmre(fmre, FIRE_WEIGHTS, device)
    vit = FrozenViTExtractor(TARGET_LAYERS).to(device).eval()

    for dataset, generator, real_dir, fake_dir in subsets:
        extract_subset(dataset, generator, real_dir, fake_dir, (fmre, vit), device)

    print(f"\nDone. Cached features under '{args.out_root}/'. "
          f"Now run:  python evaluate_report.py")


if __name__ == "__main__":
    main()
