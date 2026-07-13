import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.espcn import FrequencyMaskingModule, load_pretrained_fmre
from models.vit_extractor import FrozenViTExtractor
from utils.dataset import DeepfakeDataset

def run_extraction():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 1  # Process 1 by 1 to make saving names easy
    TARGET_LAYERS = [11, 13, 15, 17] 
    
    FIRE_WEIGHTS_PATH = "imagenet_w_adm.pt"
    TRAIN_REAL_DIR = "data/train/real"
    TRAIN_FAKE_DIR = "data/train/fake"
    
    # Create save directory
    SAVE_DIR = "processed_data/train"
    os.makedirs(f"{SAVE_DIR}/real", exist_ok=True)
    os.makedirs(f"{SAVE_DIR}/fake", exist_ok=True)

    # Dataset without training augmentations (we want clean base features)
    dataset = DeepfakeDataset(TRAIN_REAL_DIR, TRAIN_FAKE_DIR, is_train=False)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Initialize heavy models
    fmre = FrequencyMaskingModule().to(DEVICE).eval()
    fmre = load_pretrained_fmre(fmre, FIRE_WEIGHTS_PATH, DEVICE)
    vit_extractor = FrozenViTExtractor(TARGET_LAYERS).to(DEVICE).eval()

    print("--- Starting Offline Feature Extraction ---")
    
    with torch.no_grad():
        for idx, (image, label) in enumerate(tqdm(loader)):
            image = image.to(DEVICE)
            
            # 1. Generate pseudo image
            pseudo_image = fmre(image)
            
            # 2. Extract context-concatenated latent features
            # Shape: [1, 4, 2048]
            combined_features = vit_extractor(image, pseudo_image)
            
            # Remove batch dimension for cleaner saving -> Shape: [4, 2048]
            feature_tensor = combined_features.squeeze(0).cpu() 
            
            # Determine class and save path
            is_real = label.item() == 1.0
            class_folder = "real" if is_real else "fake"
            
            save_path = f"{SAVE_DIR}/{class_folder}/feature_{idx}.pt"
            torch.save(feature_tensor, save_path)

    print(f"Extraction complete! Saved all features to {SAVE_DIR}/")

if __name__ == "__main__":
    run_extraction()