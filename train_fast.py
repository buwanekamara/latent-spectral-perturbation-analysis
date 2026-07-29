import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.classifier import LSPAAttentionGatedNetwork
from utils.tensor_dataset import LatentTensorDataset


# ==================== CONFIG (edit these, then just Run) ====================
CFG_DELTA_ONLY = True   # True  -> train on Delta half only  [4, 1024]
                         # False -> train on full feature      [4, 2048]
CFG_NORMALIZE  = False   # True  -> relative-Delta normalization (divide by orig_cls norm)
CFG_EPOCHS     = 15
# ===========================================================================


def focal_bce_loss(probs, targets, gamma=2.0, eps=1e-7):
    """
    Focal loss on sigmoid probabilities.
    Down-weights easy examples so the gradient concentrates on the hard
    borderline fakes (the ones hovering at 0.55-0.60), widening the
    decision margin instead of settling for 'mostly correct'.
    """
    probs = probs.clamp(eps, 1 - eps)
    loss = -(targets * (1 - probs) ** gamma * probs.log()
             + (1 - targets) * probs ** gamma * (1 - probs).log())
    return loss.mean()


def train_fast_lspa():
    parser = argparse.ArgumentParser(description="Fast LSPA classifier training")
    # Defaults come from the CONFIG block above; CLI flags override if passed.
    parser.add_argument("--delta-only", action="store_true", default=CFG_DELTA_ONLY,
                        help="Drop the orig_cls content half; train on Delta only [4,1024]")
    parser.add_argument("--normalize", action="store_true", default=CFG_NORMALIZE,
                        help="Relative-Delta normalization (divide by orig_cls norm)")
    parser.add_argument("--epochs", type=int, default=CFG_EPOCHS)
    parser.add_argument("--out", default=None,
                        help="Output weights path (auto-named by mode if omitted)")
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32 # We can double the batch size now because memory footprint is tiny!
    EPOCHS = args.epochs
    LR = 3e-4
    NUM_LAYERS = 4
    # Feature width follows the mode: Delta-only halves it to 1024.
    FEATURE_DIM = 1024 if args.delta_only else 2048
    OUT_PATH = args.out or (
        "fast_lspa_classifier_deltaonly.pth" if args.delta_only
        else "fast_lspa_classifier.pth")

    # Paths pointing to our pre-computed tensors
    PROCESSED_REAL_DIR = "processed_data/train/real"
    PROCESSED_FAKE_DIR = "processed_data/train/fake"

    print(f"--- Starting Ultra-Fast LSPA Training on {DEVICE} ---")
    print(f"Mode: {'DELTA-ONLY [4,1024]' if args.delta_only else 'FULL [4,2048]'} "
          f"| normalize={args.normalize} | feature_dim={FEATURE_DIM}")
    print(f"Weights will be saved to: {OUT_PATH}")

    # 1. Load the pre-extracted tensors
    train_dataset = LatentTensorDataset(PROCESSED_REAL_DIR, PROCESSED_FAKE_DIR,
                                        is_train=True, normalize=args.normalize,
                                        delta_only=args.delta_only)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Load ONLY Module 3 (Trainable)
    classifier = LSPAAttentionGatedNetwork(num_layers=NUM_LAYERS,
                                           feature_dim=FEATURE_DIM).to(DEVICE)
    classifier.train()

    # 3. Optimization (focal loss replaces plain BCE - see focal_bce_loss)
    optimizer = optim.AdamW(classifier.parameters(), lr=LR, weight_decay=1e-3)

    # 4. Training Loop
    for epoch in range(EPOCHS):
        total_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, leave=True)
        for features, labels in loop:
            features, labels = features.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            # Forward pass directly into the MLP!
            preds = classifier(features)

            loss = focal_bce_loss(preds, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted_labels = (preds > 0.5).float()
            correct += (predicted_labels == labels).sum().item()
            total += labels.size(0)

            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(loss=loss.item(), acc=correct/total)

        print(f"Epoch {epoch+1} Complete | Loss: {total_loss/len(train_loader):.4f} | Acc: {correct/total:.4f}")

    torch.save(classifier.state_dict(), OUT_PATH)
    print(f"Training Complete. Classifier saved to {OUT_PATH}")

if __name__ == "__main__":
    train_fast_lspa()