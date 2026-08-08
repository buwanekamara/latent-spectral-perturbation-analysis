"""
Inference-speed benchmark:  FIRE (reconstruction)  vs  LSPA (ours).

imagenet_w_adm.pt contains the FULL FIRE model:
    vae.*                         -> Stable Diffusion VAE (encoder + decoder)
    resnet.*                      -> FIRE ResNet backend classifier
    fft_filter_module.mask_...*   -> ESPCN / FMRE (shared with LSPA)

So both pipelines can be timed end-to-end with the SAME weights, on the SAME
hardware, under the SAME protocol.

Pipelines timed (per image):
  FIRE : FMRE mask -> VAE encode+decode of image AND freq-filtered image
         (2 reconstructions) -> ResNet-50 classifier.
  LSPA : ESPCN stressor -> ViT-L(original) + ViT-L(stressed) -> attention head.

Protocol (what makes the numbers valid):
  - warm-up iterations discarded (first CUDA calls are always slow)
  - torch.cuda.synchronize() around every timed region (CUDA is async)
  - averaged over many iterations; latency (batch=1) and throughput (batch=N)
  - reports ms/image, images/sec, params, peak VRAM

Requirements: torch, torchvision, diffusers  (pip install diffusers)
Run:  python measure_speed.py
"""
import time
import argparse
import numpy as np
import torch
import torch.nn as nn

# ==================== CONFIG ====================
FIRE_WEIGHTS   = "imagenet_w_adm.pt"
LSPA_WEIGHTS   = "fast_lspa_classifier_ds.pth"   # your best model
IMG_SIZE       = 256          # common input; ViT extractor interpolates to 224 internally
TARGET_LAYERS  = [11, 13, 15, 17]
N_WARMUP       = 5
N_ITERS        = 30           # timed images for latency (enough for a stable mean)
THROUGHPUT_BATCH = 8          # batch size for throughput test
THROUGHPUT_ITERS = 15         # timed batches for throughput
FIRE_N_RECON   = 2            # FIRE reconstructs original + freq-filtered (before/after)
# ===============================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- timing helpers ----------------
def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def time_fn(fn, n_warmup=N_WARMUP, n_iters=N_ITERS):
    """Return mean, std milliseconds per call, after warm-up, CUDA-synced."""
    for _ in range(n_warmup):
        fn()
    sync()
    times = []
    for _ in range(n_iters):
        sync(); t0 = time.perf_counter()
        fn()
        sync(); times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times)), float(np.std(times))


def count_params(*modules):
    return sum(p.numel() for m in modules if m is not None
               for p in m.parameters()) / 1e6


def peak_vram_gb(fn):
    if DEVICE.type != "cuda":
        return float("nan")
    torch.cuda.reset_peak_memory_stats()
    fn(); sync()
    return torch.cuda.max_memory_allocated() / 1e9


# ---------------- LSPA pipeline ----------------
def build_lspa():
    from models.espcn import FrequencyMaskingModule, load_pretrained_fmre
    from models.vit_extractor import FrozenViTExtractor
    from models.classifier import LSPAAttentionGatedNetwork

    fmre = FrequencyMaskingModule().to(DEVICE).eval()
    fmre = load_pretrained_fmre(fmre, FIRE_WEIGHTS, DEVICE)
    vit = FrozenViTExtractor(TARGET_LAYERS).to(DEVICE).eval()
    clf = LSPAAttentionGatedNetwork(num_layers=4, feature_dim=2048).to(DEVICE).eval()
    try:
        state = torch.load(LSPA_WEIGHTS, map_location=DEVICE, weights_only=True)
        clf.load_state_dict(state, strict=False)
    except FileNotFoundError:
        print(f"[LSPA] {LSPA_WEIGHTS} not found - timing with random classifier weights "
              f"(timing is unaffected by weight values).")
    return fmre, vit, clf


def lspa_forward(models, x):
    fmre, vit, clf = models
    with torch.no_grad():
        stressed = fmre(x.float())          # ESPCN stressor (fp32)
        feats = vit(x, stressed)            # 2x ViT-L inside
        return clf(feats)


# ---------------- FIRE pipeline ----------------
def build_fire():
    """Load VAE + ResNet + FMRE from imagenet_w_adm.pt."""
    from models.espcn import FrequencyMaskingModule, load_pretrained_fmre
    import torchvision

    ckpt = torch.load(FIRE_WEIGHTS, map_location=DEVICE, weights_only=True)
    state = ckpt.get("state_dict", ckpt)

    # --- VAE (diffusers AutoencoderKL, standard SD-1.x config) ---
    from diffusers import AutoencoderKL
    vae = AutoencoderKL(
        in_channels=3, out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(128, 256, 512, 512),
        layers_per_block=2, latent_channels=4,
    )
    vae_sd = {k[len("vae."):]: v for k, v in state.items() if k.startswith("vae.")}
    missing, unexpected = vae.load_state_dict(vae_sd, strict=False)
    print(f"[FIRE] VAE loaded ({len(vae_sd)} tensors; {len(missing)} missing, "
          f"{len(unexpected)} unexpected)")
    vae = vae.to(DEVICE).eval()

    # --- ResNet-50 backend ---
    resnet = torchvision.models.resnet50(weights=None)
    res_sd = {k[len("resnet."):]: v for k, v in state.items() if k.startswith("resnet.")}
    # FIRE feeds concatenated error maps -> conv1 has >3 input channels.
    if "conv1.weight" in res_sd:
        in_ch = res_sd["conv1.weight"].shape[1]        # e.g. 6 = 2 error maps x 3
        if in_ch != 3:
            resnet.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
            print(f"[FIRE] ResNet conv1 adapted to {in_ch} input channels "
                  f"(= {in_ch // 3} reconstruction-error maps)")
            globals()["FIRE_N_RECON"] = in_ch // 3     # keep forward consistent
    if "fc.weight" in res_sd:
        out_dim = res_sd["fc.weight"].shape[0]
        resnet.fc = nn.Linear(resnet.fc.in_features, out_dim)
    missing, unexpected = resnet.load_state_dict(res_sd, strict=False)
    print(f"[FIRE] ResNet loaded ({len(res_sd)} tensors; {len(missing)} missing, "
          f"{len(unexpected)} unexpected)")
    resnet = resnet.to(DEVICE).eval()

    # --- FMRE (shared ESPCN) ---
    fmre = FrequencyMaskingModule().to(DEVICE).eval()
    fmre = load_pretrained_fmre(fmre, FIRE_WEIGHTS, DEVICE)

    return fmre, vae, resnet


def fire_forward(models, x):
    fmre, vae, resnet = models
    with torch.no_grad():
        filtered = fmre(x.float())                      # frequency-filtered image
        inputs = [x, filtered][:FIRE_N_RECON]
        errors = []
        for img in inputs:
            lat = vae.encode(img.float()).latent_dist.mean   # encode
            rec = vae.decode(lat).sample                     # decode
            errors.append((img - rec).abs())                 # reconstruction error
        # FIRE concatenates the error maps along channels (matches conv1 in-channels)
        err = torch.cat(errors, dim=1)
        return resnet(err)


# ---------------- benchmark driver ----------------
def bench(name, build_fn, forward_fn):
    print(f"\n=== Building {name} ===", flush=True)
    try:
        models = build_fn()
    except Exception as e:
        print(f"[{name}] could not build: {e}")
        return None

    x1 = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    xb = torch.randn(THROUGHPUT_BATCH, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)

    print(f"[{name}] timing latency ({N_WARMUP} warmup + {N_ITERS} iters)...", flush=True)
    t0 = time.time()
    lat_mean, lat_std = time_fn(lambda: forward_fn(models, x1))
    print(f"[{name}]   -> {lat_mean:.2f} ms/img  (took {time.time()-t0:.0f}s)", flush=True)

    print(f"[{name}] timing throughput (batch={THROUGHPUT_BATCH})...", flush=True)
    tb_mean, _ = time_fn(lambda: forward_fn(models, xb),
                         n_warmup=3, n_iters=THROUGHPUT_ITERS)
    throughput = THROUGHPUT_BATCH / (tb_mean / 1000.0)

    print(f"[{name}] measuring peak VRAM...", flush=True)
    vram = peak_vram_gb(lambda: forward_fn(models, x1))
    params = count_params(*models)
    print(f"[{name}] done.", flush=True)

    return dict(name=name, lat_mean=lat_mean, lat_std=lat_std,
                throughput=throughput, params=params, vram=vram)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--iters", type=int, default=N_ITERS)
    args = parser.parse_args()
    globals()["IMG_SIZE"] = args.img_size
    globals()["N_ITERS"] = args.iters

    print(f"Device: {DEVICE} | image {IMG_SIZE}x{IMG_SIZE} | "
          f"{N_ITERS} timed iters | warmup {N_WARMUP}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results = []
    r_lspa = bench("LSPA (Ours)", build_lspa, lspa_forward)
    r_fire = bench("FIRE (recon+ResNet)", build_fire, fire_forward)
    results = [r for r in (r_fire, r_lspa) if r]

    print("\n==================================================================")
    print(f"{'Method':<24}{'ms/img':>10}{'img/s':>10}{'Params(M)':>12}{'VRAM(GB)':>10}")
    print("------------------------------------------------------------------")
    for r in results:
        print(f"{r['name']:<24}{r['lat_mean']:>8.2f}±{r['lat_std']:<1.0f}"
              f"{r['throughput']:>9.1f}{r['params']:>12.1f}{r['vram']:>10.2f}")
    print("==================================================================")

    if len(results) == 2:
        f, l = (r for r in results if "FIRE" in r["name"]), None
        fire = next(r for r in results if "FIRE" in r["name"])
        lspa = next(r for r in results if "LSPA" in r["name"])
        speedup = fire["lat_mean"] / lspa["lat_mean"]
        verb = "faster" if speedup > 1 else "slower"
        print(f"\nLSPA is {speedup:.2f}x {verb} than FIRE per image "
              f"({lspa['lat_mean']:.2f} vs {fire['lat_mean']:.2f} ms).")
    print("\nReport these numbers alongside your accuracy/AUC table.")


if __name__ == "__main__":
    main()
