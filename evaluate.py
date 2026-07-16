import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np

# Import your model and dataset
from models.classifier import LSPAAttentionGatedNetwork
from utils.tensor_dataset import LatentTensorDataset

def evaluate_model():
    # ---------------- Configuration ----------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    NUM_LAYERS = 4
    
    # Path to your saved weights from the training phase
    MODEL_WEIGHTS_PATH = "fast_lspa_classifier.pth" 
    
    # Paths to your extracted validation tensors
    VAL_REAL_DIR = "processed_data/val/real"
    VAL_FAKE_DIR = "processed_data/val/fake"

    print(f"--- Starting LSPA Evaluation on {DEVICE} ---")
    print(f"Loading weights from: {MODEL_WEIGHTS_PATH}")

    # 1. Load the Validation Dataset
    val_dataset = LatentTensorDataset(VAL_REAL_DIR, VAL_FAKE_DIR, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Initialize the Model and Load Weights
    classifier = LSPAAttentionGatedNetwork(num_layers=NUM_LAYERS).to(DEVICE)
    
    try:
        classifier.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
        print("✅ Model weights loaded successfully.")
    except FileNotFoundError:
        print(f"❌ Error: Could not find '{MODEL_WEIGHTS_PATH}'. Please ensure the path is correct.")
        return

    # Set model to evaluation mode (turns off Dropout)
    classifier.eval()

    # 3. Inference Loop
    all_preds = []
    all_labels = []

    print("\nRunning inference on validation set...")
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            
            # Forward pass
            preds = classifier(features)
            
            # Store probabilities and true labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 4. Calculate Metrics
    # Convert probabilities to hard classes (0 or 1) using a 0.5 threshold
    pred_classes = [1 if p > 0.5 else 0 for p in all_preds]
    
    acc = accuracy_score(all_labels, pred_classes)
    prec = precision_score(all_labels, pred_classes, zero_division=0)
    rec = recall_score(all_labels, pred_classes, zero_division=0)
    f1 = f1_score(all_labels, pred_classes, zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc = 0.0 # Occurs if validation set only has one class
        
    cm = confusion_matrix(all_labels, pred_classes)

    # 5. Print Academic Report
    print("\n==========================================")
    print("      LSPA VALIDATION METRICS REPORT      ")
    print("==========================================")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%  <- (How many 'Reals' were actually Real?)")
    print(f"Recall:    {rec * 100:.2f}%  <- (How many actual 'Reals' did we find?)")
    print(f"F1-Score:  {f1 * 100:.2f}%  <- (Overall balance of the model)")
    print(f"AUC-ROC:   {auc * 100:.2f}%  <- (Separation capacity between classes)")
    print("------------------------------------------")
    print("Confusion Matrix:")
    print(f"True Fake (TN): {cm[0][0]} | False Real (FP): {cm[0][1]}")
    print(f"False Fake (FN): {cm[1][0]} | True Real (TP): {cm[1][1]}")
    print("==========================================\n")

if __name__ == "__main__":
    evaluate_model()