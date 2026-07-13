import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.classifier import LSPAGatedNetwork
from utils.tensor_dataset import LatentTensorDataset

def train_fast_lspa():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32 # We can double the batch size now because memory footprint is tiny!
    EPOCHS = 15
    LR = 3e-4
    NUM_LAYERS = 4
    
    # Paths pointing to our pre-computed tensors
    PROCESSED_REAL_DIR = "processed_data/train/real"
    PROCESSED_FAKE_DIR = "processed_data/train/fake"
    
    print(f"--- Starting Ultra-Fast LSPA Training on {DEVICE} ---")

    # 1. Load the pre-extracted tensors
    train_dataset = LatentTensorDataset(PROCESSED_REAL_DIR, PROCESSED_FAKE_DIR, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Load ONLY Module 3 (Trainable)
    classifier = LSPAGatedNetwork(num_layers=NUM_LAYERS).to(DEVICE)
    classifier.train()

    # 3. Optimization
    criterion = nn.BCELoss()
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
            
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted_labels = (preds > 0.5).float()
            correct += (predicted_labels == labels).sum().item()
            total += labels.size(0)

            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(loss=loss.item(), acc=correct/total)

        print(f"Epoch {epoch+1} Complete | Loss: {total_loss/len(train_loader):.4f} | Acc: {correct/total:.4f}")

    torch.save(classifier.state_dict(), "fast_lspa_classifier.pth")
    print("Training Complete. Classifier saved successfully.")

if __name__ == "__main__":
    train_fast_lspa()