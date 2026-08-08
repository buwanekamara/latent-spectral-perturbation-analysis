"""
Detailed evaluation report for LSPA across all model-wise evaluation sets.

Reads cached features from processed_eval/<dataset>/<generator>/{real,fake}
(produced by extract_eval_features.py), runs the trained classifier on each
generator separately, and writes:

  - console summary
  - evaluation_report.md   (human-readable tables)
  - evaluation_results.csv (for your own plotting)

Metrics per generator:
  AUC (threshold-free, primary), EER, Accuracy@0.5,
  calibrated accuracy (leakage-free: threshold picked on a held-out half,
  measured on the other half, averaged over many splits),
  plus a "deployment" accuracy using ONE threshold frozen on the in-domain
  set and applied everywhere (mimics real use).
"""
import os
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (roc_auc_score, roc_curve, accuracy_score,
                             precision_score, recall_score, f1_score)

from models.classifier import LSPAAttentionGatedNetwork
from utils.tensor_dataset import LatentTensorDataset

# ==================== CONFIG (match how the weights were trained) ==========
PROC_ROOT        = "processed_eval"
NORMALIZE        = True
DELTA_ONLY       = False
DEEP_SUPERVISION = True          # only affects which weights file is loaded
WEIGHTS          = None          # None -> auto-name by the flags above
# Which generator counts as "seen" (the model trained on it):
SEEN_GENERATORS  = {"adm"}
# The in-domain subset used to freeze the deployment threshold:
INDOMAIN_KEY     = ("imagenet", "adm")
CALIB_RUNS       = 20
CALIB_FRAC       = 0.5
# ===========================================================================


def auto_weights():
    if WEIGHTS:
        return WEIGHTS
    name = "fast_lspa_classifier"
    if DELTA_ONLY:
        name += "_deltaonly"
    if DEEP_SUPERVISION:
        name += "_ds"
    return name + ".pth"


def load_model(device, feature_dim):
    model = LSPAAttentionGatedNetwork(num_layers=4, feature_dim=feature_dim).to(device)
    state = torch.load(auto_weights(), map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)   # ignore training-only aux heads
    model.eval()
    return model


def score_subset(model, real_dir, fake_dir, device):
    ds = LatentTensorDataset(real_dir, fake_dir, is_train=False,
                             normalize=NORMALIZE, delta_only=DELTA_ONLY)
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    probs, labels = [], []
    with torch.no_grad():
        for feats, labs in loader:
            p = model(feats.to(device))
            probs.extend(p.cpu().numpy().ravel())
            labels.extend(labs.numpy().ravel())
    return np.array(probs), np.array(labels)


def youden_threshold(labels, probs):
    fpr, tpr, thr = roc_curve(labels, probs)
    finite = np.isfinite(thr)
    fpr, tpr, thr = fpr[finite], tpr[finite], thr[finite]
    return float(thr[int(np.argmax(tpr - fpr))])


def eer_value(labels, probs):
    fpr, tpr, thr = roc_curve(labels, probs)
    i = int(np.argmin(np.abs(fpr - (1 - tpr))))
    return float((fpr[i] + (1 - tpr[i])) / 2)


def calibrated_accuracy(labels, probs, runs=CALIB_RUNS, frac=CALIB_FRAC):
    """Leakage-free: threshold from one half, measured on the other, averaged."""
    accs = []
    for seed in range(runs):
        rng = np.random.RandomState(seed)
        calib, test = [], []
        for cls in (0, 1):
            idx = np.where(labels == cls)[0]
            rng.shuffle(idx)
            cut = int(len(idx) * frac)
            test.append(idx[:cut]); calib.append(idx[cut:])
        calib, test = np.concatenate(calib), np.concatenate(test)
        thr = youden_threshold(labels[calib], probs[calib])
        preds = (probs[test] >= thr).astype(int)
        accs.append(accuracy_score(labels[test], preds))
    return np.mean(accs) * 100, np.std(accs) * 100


def metrics_at(labels, probs, thr):
    preds = (probs >= thr).astype(int)
    return dict(
        acc=accuracy_score(labels, preds) * 100,
        prec=precision_score(labels, preds, zero_division=0) * 100,
        rec=recall_score(labels, preds, zero_division=0) * 100,
        f1=f1_score(labels, preds, zero_division=0) * 100,
    )


def discover(proc_root):
    subsets = []
    for dataset in sorted(os.listdir(proc_root)):
        dpath = os.path.join(proc_root, dataset)
        if not os.path.isdir(dpath):
            continue
        for generator in sorted(os.listdir(dpath)):
            gpath = os.path.join(dpath, generator)
            r, f = os.path.join(gpath, "real"), os.path.join(gpath, "fake")
            if os.path.isdir(r) and os.path.isdir(f):
                subsets.append((dataset, generator, r, f))
    return subsets


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_dim = 1024 if DELTA_ONLY else 2048

    if not os.path.isdir(PROC_ROOT):
        print(f"'{PROC_ROOT}/' not found. Run extract_eval_features.py first.")
        return

    print(f"Loading weights: {auto_weights()} | normalize={NORMALIZE} "
          f"| feature_dim={feature_dim}")
    model = load_model(device, feature_dim)

    subsets = discover(PROC_ROOT)
    if not subsets:
        print(f"No cached subsets under '{PROC_ROOT}/'. Run extract_eval_features.py.")
        return

    # 1. Score every generator subset once.
    rows = []
    scores = {}
    for dataset, gen, r, f in subsets:
        probs, labels = score_subset(model, r, f, device)
        scores[(dataset, gen)] = (probs, labels)
        if (labels == 1).sum() == 0 or (labels == 0).sum() == 0:
            print(f"  [warn] {dataset}/{gen}: only one class, skipping")
            continue
        auc = roc_auc_score(labels, probs) * 100
        eer = eer_value(labels, probs) * 100
        acc05 = metrics_at(labels, probs, 0.5)
        cal_acc, cal_std = calibrated_accuracy(labels, probs)
        rows.append(dict(
            dataset=dataset, generator=gen,
            n_real=int((labels == 1).sum()), n_fake=int((labels == 0).sum()),
            seen=("seen" if gen in SEEN_GENERATORS else "unseen"),
            auc=auc, eer=eer, acc05=acc05["acc"],
            cal_acc=cal_acc, cal_std=cal_std,
            prec=acc05["prec"], rec=acc05["rec"], f1=acc05["f1"],
        ))

    # 2. Deployment threshold: frozen on the in-domain set, applied to all.
    deploy_thr = None
    if INDOMAIN_KEY in scores:
        pl, ll = scores[INDOMAIN_KEY]
        if (ll == 1).sum() and (ll == 0).sum():
            deploy_thr = youden_threshold(ll, pl)
    for row in rows:
        if deploy_thr is not None:
            p, l = scores[(row["dataset"], row["generator"])]
            row["acc_deploy"] = metrics_at(l, p, deploy_thr)["acc"]
        else:
            row["acc_deploy"] = float("nan")

    write_report(rows, deploy_thr)


def write_report(rows, deploy_thr):
    lines = []
    def out(s=""):
        print(s); lines.append(s)

    out("# LSPA Detailed Evaluation Report")
    out(f"\nWeights: `{auto_weights()}`  |  normalize={NORMALIZE}  |  "
        f"deep_supervision={DEEP_SUPERVISION}")
    if deploy_thr is not None:
        out(f"Deployment threshold (frozen on {INDOMAIN_KEY[0]}/{INDOMAIN_KEY[1]}): "
            f"{deploy_thr:.4f}")

    # Group by dataset
    datasets = sorted(set(r["dataset"] for r in rows))
    for dataset in datasets:
        drows = [r for r in rows if r["dataset"] == dataset]
        out(f"\n## {dataset}\n")
        out("| Generator | Seen | #real | #fake | AUC % | EER % | Acc@0.5 % | "
            "Cal.Acc % (±std) | Deploy Acc % | Prec % | Rec % | F1 % |")
        out("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in drows:
            out(f"| {r['generator']} | {r['seen']} | {r['n_real']} | {r['n_fake']} | "
                f"{r['auc']:.2f} | {r['eer']:.2f} | {r['acc05']:.2f} | "
                f"{r['cal_acc']:.2f} ± {r['cal_std']:.2f} | {r['acc_deploy']:.2f} | "
                f"{r['prec']:.2f} | {r['rec']:.2f} | {r['f1']:.2f} |")

    # Summary averages
    def avg(rs, key):
        return np.mean([r[key] for r in rs]) if rs else float("nan")

    unseen = [r for r in rows if r["seen"] == "unseen"]
    seen = [r for r in rows if r["seen"] == "seen"]
    gen_bench = [r for r in rows if r["dataset"] == "tiny_genimage"]

    out("\n## Summary\n")
    out("| Group | #subsets | Avg AUC % | Avg Cal.Acc % | Avg Deploy Acc % |")
    out("|---|---|---|---|---|")
    for name, rs in [("Seen (in-domain)", seen),
                     ("Unseen (generalization)", unseen),
                     ("tiny_genimage (all)", gen_bench),
                     ("ALL subsets", rows)]:
        out(f"| {name} | {len(rs)} | {avg(rs,'auc'):.2f} | "
            f"{avg(rs,'cal_acc'):.2f} | {avg(rs,'acc_deploy'):.2f} |")

    out("\n**For the thesis:** lead with AUC (threshold-free). Report the "
        "generalization row (unseen average) as the headline number, and the "
        "in-domain seen row as a one-line 'fits training distribution' note.")

    # Write files
    with open("evaluation_report.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    with open("evaluation_results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nSaved: evaluation_report.md  and  evaluation_results.csv")


if __name__ == "__main__":
    main()
