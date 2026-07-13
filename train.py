import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.espcn import FrequencyMaskingModule, load_pretrained_fmre
from models.vit_extractor import FrozenViTExtractor
from models.classifier import LSPAGatedNetwork
from utils.dataset import DeepfakeDataset

def train_lspa():
    # ---------------- Configuration ----------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 16
    EPOCHS = 10
    LR = 1e-4
    
    # Layer Ablation target - change this to [17, 19, 21, 23] for deep testing
    TARGET_LAYERS = [11, 13, 15, 17] 
    
    # Paths (Update these to your local or mounted drive paths)
    FIRE_WEIGHTS_PATH = "imagenet_w_adm.pt"
    TRAIN_REAL_DIR = "data/train/real"
    TRAIN_FAKE_DIR = "data/train/fake"
    
    print(f"--- Starting LSPA Training on {DEVICE} ---")

    # ---------------- Data Loaders ----------------
    train_dataset = DeepfakeDataset(TRAIN_REAL_DIR, TRAIN_FAKE_DIR, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # ---------------- Initialize Modules ----------------
    # Module 1 (Frozen)
    fmre = FrequencyMaskingModule().to(DEVICE)
    fmre = load_pretrained_fmre(fmre, FIRE_WEIGHTS_PATH, DEVICE)
    fmre.eval()
    
    # Module 2 (Frozen)
    vit_extractor = FrozenViTExtractor(TARGET_LAYERS).to(DEVICE)
    
    # Module 3 (Trainable)
    classifier = LSPAGatedNetwork(num_layers=len(TARGET_LAYERS)).to(DEVICE)
    classifier.train()

    # ---------------- Optimization ----------------
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(classifier.parameters(), lr=LR, weight_decay=1e-4)

    # ---------------- Training Loop ----------------
    for epoch in range(EPOCHS):
        total_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, leave=True)
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()

            with torch.no_grad():
                # 1. Stress the image (Mask frequencies)
                pseudo_images = fmre(images)
                
                # 2. Extract Latent Features [Original + Delta]
                combined_features = vit_extractor(images, pseudo_images)
                
            # 3. Forward pass through trainable MLP
            preds = classifier(combined_features)
            
            # 4. Calculate Loss & Backprop
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            # Metrics
            total_loss += loss.item()
            predicted_labels = (preds > 0.5).float()
            correct += (predicted_labels == labels).sum().item()
            total += labels.size(0)

            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(loss=loss.item(), acc=correct/total)

        epoch_acc = correct / total
        print(f"End of Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Acc: {epoch_acc:.4f}")

    # Save the trained model weights
    torch.save(classifier.state_dict(), f"lspa_classifier_layers_{TARGET_LAYERS[0]}_to_{TARGET_LAYERS[-1]}.pth")
    print("Training Complete. Model Saved.")

if __name__ == "__main__":
    train_lspa()