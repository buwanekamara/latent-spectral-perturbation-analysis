"""
Feature extraction for the model-wise evaluation sets (standalone).

Separate from extract_eval_features.py / extract_features.py / extract_val_features.py
so your other extraction scripts stay untouched.

Walks:  Evaluation data set/<dataset>/<generator>/{real,fake}/*.png
Caches: processed_eval/<dataset>/<generator>/{real,fake}/feature_N.pt   ([4,2048])

Idempotent: a generator already cached is skipped, so re-running is safe.
Then run evaluate_report.py to produce the detailed report.
"""
import os
import time
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.espcn import FrequencyMaskingModule, load_pretrained_fmre
from models.vit_extractor import FrozenViTExtractor
from utils.dataset import DeepfakeDataset

# ==================== CONFIG ====================
EVAL_ROOT     = "Evaluation data set"   # input root (your folders)
OUT_ROOT      = "processed_eval"        # cached-feature root (created here)
FIRE_WEIGHTS  = "imagenet_w_adm.pt"
TARGET_LAYERS = [11, 13, 15, 17]
IMG_EXTS      = (".png", ".jpg", ".jpeg", ".webp")
# ===============================================


def has_images(d):
    return os.path.isdir(d) and any(f.lower().endswith(IMG_EXTS) for f in os.listdir(d))


def has_pt(d):
    return os.path.isdir(d) and any(f.endswith(".pt") for f in os.listdir(d))


def discover_subsets(root):
    """Return [(dataset, generator, real_dir, fake_dir), ...]."""
    subsets = []
    if not os.path.isdir(root):
        return subsets
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


def count_images(d):
    return sum(1 for f in os.listdir(d) if f.lower().endswith(IMG_EXTS)) if os.path.isdir(d) else 0


def extract_subset(dataset, generator, real_dir, fake_dir, fmre, vit, device,
                   out_root, position="", real_count=0, fake_count=0, elapsed_start=None):
    out_real = os.path.join(out_root, dataset, generator, "real")
    out_fake = os.path.join(out_root, dataset, generator, "fake")

    if has_pt(out_real) and has_pt(out_fake):
        print(f"{position} [SKIP] {dataset}/{generator} - already cached\n", flush=True)
        return 0

    os.makedirs(out_real, exist_ok=True)
    os.makedirs(out_fake, exist_ok=True)

    total = real_count + fake_count
    print(f"{position} [START] {dataset}/{generator}  "
          f"({real_count} real + {fake_count} fake = {total} images)", flush=True)

    loader = DataLoader(DeepfakeDataset(real_dir, fake_dir, is_train=False),
                        batch_size=1, shuffle=False, num_workers=2)

    t0 = time.time()
    n_real = n_fake = 0
    with torch.no_grad():
        # tqdm shows a live bar; ncols keeps it tidy, mininterval throttles updates
        for idx, (image, label) in enumerate(
                tqdm(loader, desc=f"  {dataset}/{generator}", ncols=90, mininterval=0.5)):
            image = image.to(device)
            # FFT stressor in fp32 (half precision overflows the magnitudes -> NaN)
            pseudo = fmre(image.float())
            feats = vit(image, pseudo).squeeze(0).cpu()          # [4, 2048]
            if label.item() == 1.0:
                torch.save(feats, os.path.join(out_real, f"feature_{idx}.pt"))
                n_real += 1
            else:
                torch.save(feats, os.path.join(out_fake, f"feature_{idx}.pt"))
                n_fake += 1

    dt = time.time() - t0
    rate = total / dt if dt > 0 else 0
    print(f"{position} [DONE]  {dataset}/{generator} - saved {n_real} real + {n_fake} fake "
          f"in {dt:.1f}s ({rate:.1f} img/s)\n", flush=True)
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", default=EVAL_ROOT)
    parser.add_argument("--out-root", default=OUT_ROOT)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subsets = discover_subsets(args.eval_root)
    if not subsets:
        print(f"No <dataset>/<generator>/{{real,fake}} folders found under '{args.eval_root}'.")
        return

    # Pre-count images so we can show total workload up front.
    grand_total = 0
    print(f"Found {len(subsets)} generator subsets on {device}:")
    counts = []
    for d, g, r, f in subsets:
        rc, fc = count_images(r), count_images(f)
        counts.append((rc, fc))
        grand_total += rc + fc
        print(f"  - {d}/{g}: {rc} real + {fc} fake")
    print(f"Total images to process: {grand_total}\n")

    print("Loading frozen models (ESPCN stressor + CLIP ViT-L)...", flush=True)
    fmre = FrequencyMaskingModule().to(device).eval()
    fmre = load_pretrained_fmre(fmre, FIRE_WEIGHTS, device)
    vit = FrozenViTExtractor(TARGET_LAYERS).to(device).eval()
    print("Models ready. Starting extraction.\n", flush=True)

    t_start = time.time()
    done = 0
    for i, (dataset, generator, real_dir, fake_dir) in enumerate(subsets, 1):
        rc, fc = counts[i - 1]
        position = f"[{i}/{len(subsets)}]"
        done += extract_subset(dataset, generator, real_dir, fake_dir,
                               fmre, vit, device, args.out_root,
                               position=position, real_count=rc, fake_count=fc)

    mins = (time.time() - t_start) / 60
    print(f"ALL DONE. Cached features under '{args.out_root}/' "
          f"(processed {done} new images in {mins:.1f} min).")
    print("Now run:  python evaluate_report.py")


if __name__ == "__main__":
    main()
