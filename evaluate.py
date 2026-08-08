import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)

from models.classifier import LSPAAttentionGatedNetwork
from utils.tensor_dataset import LatentTensorDataset


# ==================== CONFIG (edit these, then just Run) ====================
CFG_DELTA_ONLY        = False   # MUST match the mode the weights were trained with
CFG_NORMALIZE         = True    # MUST match training
CFG_DEEP_SUPERVISION  = True    # MUST match training (only affects which file is loaded)
# Weights auto-select by mode below; override with CFG_WEIGHTS if you want.
CFG_WEIGHTS    = None     # None -> auto-named by the mode flags above
CFG_VAL_REAL   = "processed_data/val/real"
CFG_VAL_FAKE   = "processed_data/val/fake"
CFG_CALIBRATE  = True     # True -> also report leakage-free calibrated accuracy
CFG_CALIB_FRAC = 0.5      # fraction of the set used to PICK the threshold
CFG_CALIB_RUNS = 20       # repeated random splits, results averaged
# ===========================================================================


def group_key(path):
    """
    TTA support: crops of the same image are named  <imgid>_crop<k>.pt
    and get averaged into one score. Legacy files (feature_<idx>.pt)
    have no '_crop' part, so each file is its own group.
    """
    name = os.path.splitext(os.path.basename(path))[0]
    # Include the parent folder (real/fake) - basenames repeat across classes!
    return os.path.join(os.path.dirname(path), name.split("_crop")[0])


def compute_metrics(labels, probs, threshold):
    # Use >= to match sklearn's roc_curve convention. With < the reported
    # metrics collapse to 0 whenever a threshold equals a score exactly -
    # which happens as soon as the sigmoid saturates (real scores hit 1.0).
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "cm": cm,
    }


def youden_threshold(labels, probs):
    """Best-balanced threshold (max TPR-FPR) on the GIVEN data."""
    fpr, tpr, thr = roc_curve(labels, probs)
    finite = np.isfinite(thr)
    fpr, tpr, thr = fpr[finite], tpr[finite], thr[finite]
    return float(thr[int(np.argmax(tpr - fpr))])


def stratified_split(labels, test_frac, rng):
    """Split indices keeping the real/fake ratio in both halves."""
    calib_idx, test_idx = [], []
    for cls in (0, 1):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        cut = int(len(idx) * test_frac)
        test_idx.append(idx[:cut])
        calib_idx.append(idx[cut:])
    return np.concatenate(calib_idx), np.concatenate(test_idx)


def calibrated_report(labels, probs, runs, test_frac):
    """
    Leakage-free accuracy: pick the threshold on a CALIBRATION half, then
    measure on the untouched TEST half. Repeat over many random splits and
    average, so the reported number does not depend on one lucky split and
    was never tuned on the data it is measured on.
    """
    accs, precs, recs, f1s = [], [], [], []
    for seed in range(runs):
        rng = np.random.RandomState(seed)
        calib_idx, test_idx = stratified_split(labels, test_frac, rng)
        thr = youden_threshold(labels[calib_idx], probs[calib_idx])   # chosen on calib
        m = compute_metrics(labels[test_idx], probs[test_idx], thr)   # measured on test
        accs.append(m["accuracy"]); precs.append(m["precision"])
        recs.append(m["recall"]);   f1s.append(m["f1"])

    def ms(a):
        a = np.array(a) * 100
        return a.mean(), a.std()

    print("\n==========================================")
    print("   CALIBRATED (LEAKAGE-FREE) METRICS      ")
    print(f"   threshold picked on {int((1-test_frac)*100)}% held-out, "
          f"measured on the other {int(test_frac*100)}%, {runs} splits")
    print("==========================================")
    for name, vals in [("Accuracy", accs), ("Precision", precs),
                       ("Recall", recs), ("F1-Score", f1s)]:
        mean, std = ms(vals)
        print(f"{name:10s}: {mean:.2f}% +/- {std:.2f}")
    print("==========================================")
    print("Report these numbers (with AUC) - the threshold never saw the test data.\n")


def print_report(title, m):
    print(f"\n--- {title} (threshold = {m['threshold']:.4f}) ---")
    print(f"Accuracy:  {m['accuracy'] * 100:.2f}%")
    print(f"Precision: {m['precision'] * 100:.2f}%  <- (How many predicted 'Reals' were actually Real?)")
    print(f"Recall:    {m['recall'] * 100:.2f}%  <- (How many actual 'Reals' did we find?)")
    print(f"F1-Score:  {m['f1'] * 100:.2f}%")
    cm = m["cm"]
    print("Confusion Matrix:")
    print(f"True Fake (TN): {cm[0][0]} | False Real (FP): {cm[0][1]}")
    print(f"False Fake (FN): {cm[1][0]} | True Real (TP): {cm[1][1]}")


def evaluate_model():
    parser = argparse.ArgumentParser(description="LSPA evaluation with threshold sweep + TTA")
    # Defaults come from the CONFIG block above; CLI flags override if passed.
    parser.add_argument("--real-dir", default=CFG_VAL_REAL)
    parser.add_argument("--fake-dir", default=CFG_VAL_FAKE)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Optional extra manual threshold to report")
    parser.add_argument("--normalize", action="store_true", default=CFG_NORMALIZE,
                        help="Enable relative-Delta feature normalization "
                             "(must match how the weights were trained)")
    parser.add_argument("--delta-only", action="store_true", default=CFG_DELTA_ONLY,
                        help="Use Delta half only [4,1024] (must match training)")
    parser.add_argument("--deep-supervision", action="store_true", default=CFG_DEEP_SUPERVISION,
                        help="Load a deep-supervision-trained file (aux heads ignored at eval)")
    parser.add_argument("--no-calibrate", action="store_true", default=not CFG_CALIBRATE,
                        help="Skip the leakage-free calibrated report")
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 64
    NUM_LAYERS = 4
    FEATURE_DIM = 1024 if args.delta_only else 2048
    # Auto-select weights file to match the mode, unless one was given.
    if args.weights is None:
        name = "fast_lspa_classifier"
        if args.delta_only:
            name += "_deltaonly"
        if args.deep_supervision:
            name += "_ds"
        args.weights = CFG_WEIGHTS or (name + ".pth")

    print(f"--- Starting LSPA Evaluation on {DEVICE} ---")
    print(f"Weights: {args.weights} | Real: {args.real_dir} | Fake: {args.fake_dir}")
    print(f"Mode: {'DELTA-ONLY [4,1024]' if args.delta_only else 'FULL [4,2048]'} "
          f"| normalization: {'ON' if args.normalize else 'OFF'}")

    # 1. Dataset (order preserved: shuffle=False)
    val_dataset = LatentTensorDataset(args.real_dir, args.fake_dir,
                                      is_train=False, normalize=args.normalize,
                                      delta_only=args.delta_only)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Model
    classifier = LSPAAttentionGatedNetwork(num_layers=NUM_LAYERS,
                                           feature_dim=FEATURE_DIM).to(DEVICE)
    try:
        state = torch.load(args.weights, map_location=DEVICE, weights_only=True)
        # strict=False: a deep-supervision file carries aux_heads.* keys that the
        # eval model doesn't have (aux heads are training-only). Everything the
        # eval model needs (projection/attention/classifier) still loads.
        missing, unexpected = classifier.load_state_dict(state, strict=False)
        aux_ignored = [k for k in unexpected if k.startswith("aux_heads")]
        real_missing = [k for k in missing]
        if real_missing:
            print(f"WARNING: missing weights not found in file: {real_missing}")
        if aux_ignored:
            print(f"Model weights loaded (ignored {len(aux_ignored)} training-only aux-head tensors).")
        else:
            print("Model weights loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Could not find '{args.weights}'.")
        return
    classifier.eval()

    # 3. Inference (model ends in Sigmoid -> outputs are probabilities of 'Real')
    all_probs, all_labels = [], []
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(DEVICE)
            probs = classifier(features)
            all_probs.extend(probs.cpu().numpy().ravel())
            all_labels.extend(labels.numpy().ravel())
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # 4. TTA aggregation: average crop scores per source image
    keys = [group_key(p) for p in val_dataset.file_paths]
    grouped = {}
    for k, prob, lab in zip(keys, all_probs, all_labels):
        grouped.setdefault(k, [[], lab])[0].append(prob)
    probs = np.array([np.mean(v[0]) for v in grouped.values()])
    labels = np.array([v[1] for v in grouped.values()])
    n_crops = len(all_probs) / max(len(probs), 1)
    if n_crops > 1.01:
        print(f"TTA detected: {len(all_probs)} crops -> {len(probs)} images "
              f"(~{n_crops:.1f} crops/image, scores averaged)")

    # Sanity guard: NaN/inf scores (usually corrupt cached features)
    bad = ~np.isfinite(all_probs)
    if bad.any():
        print(f"\nERROR: {bad.sum()} of {len(all_probs)} crop scores are NaN/inf.")
        print("Offending cached feature files (first 10):")
        for p in np.array(val_dataset.file_paths)[bad][:10]:
            print(f"  {p}")
        print("Cause is almost always fp16 extraction overflowing the FFT stressor.")
        print("Re-extract without --fp16, then evaluate again. Aborting.")
        return

    # Sanity guard: both classes must survive grouping
    n_real, n_fake = int((labels == 1).sum()), int((labels == 0).sum())
    print(f"Grouped samples: {n_real} real | {n_fake} fake")
    if n_real == 0 or n_fake == 0:
        print("ERROR: only one class present after grouping - check your "
              "--real-dir/--fake-dir paths and file naming. Aborting.")
        return

    # 5. Threshold-independent metrics
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")

    fpr, tpr, thresholds = roc_curve(labels, probs)

    # sklearn prepends an artificial threshold (inf) for the (0,0) ROC point.
    # It is not a usable operating point, so drop it before searching.
    finite = np.isfinite(thresholds)
    fpr, tpr, thresholds = fpr[finite], tpr[finite], thresholds[finite]

    # Youden's J: threshold maximizing TPR - FPR (best balanced operating point)
    j_idx = int(np.argmax(tpr - fpr))
    youden_thr = float(thresholds[j_idx])

    # Equal Error Rate: point where FPR == 1 - TPR (FNR)
    eer_idx = int(np.argmin(np.abs(fpr - (1 - tpr))))
    eer = float((fpr[eer_idx] + (1 - tpr[eer_idx])) / 2)
    eer_thr = float(thresholds[eer_idx])

    print("\n==========================================")
    print("      LSPA VALIDATION METRICS REPORT      ")
    print("==========================================")
    print(f"AUC-ROC: {auc * 100:.2f}%  (threshold-independent separation)")
    print(f"EER:     {eer * 100:.2f}%  (at threshold {eer_thr:.4f})")
    print(f"Score stats  | Real: mean {probs[labels == 1].mean():.3f}"
          f" | Fake: mean {probs[labels == 0].mean():.3f}")

    print("\n--- Thresholds picked on THIS set (informational, NOT for thesis: leaky) ---")
    print_report("Default", compute_metrics(labels, probs, 0.5))
    print_report("Youden's J optimal", compute_metrics(labels, probs, youden_thr))
    print_report("EER operating point", compute_metrics(labels, probs, eer_thr))
    if args.threshold is not None:
        print_report("Manual", compute_metrics(labels, probs, args.threshold))
    print("==========================================")

    # Leakage-free calibrated accuracy (the number to actually report)
    if not args.no_calibrate:
        calibrated_report(labels, probs, runs=CFG_CALIB_RUNS, test_frac=CFG_CALIB_FRAC)


if __name__ == "__main__":
    evaluate_model()
