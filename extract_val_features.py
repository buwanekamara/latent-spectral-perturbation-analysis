import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.espcn import FrequencyMaskingModule, load_pretrained_fmre
from models.vit_extractor import FrozenViTExtractor
from utils.dataset import DeepfakeDataset

def run_val_extraction():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 1 
    TARGET_LAYERS = [11, 13, 15, 17] 
    
    FIRE_WEIGHTS_PATH = "imagenet_w_adm.pt"
    
    # 1. Point specifically to your validation folders
    VAL_REAL_DIR = "data/val/real"
    VAL_FAKE_DIR = "data/val/fake"
    
    # Create the processed save directories
    SAVE_DIR = "processed_data/val"
    os.makedirs(f"{SAVE_DIR}/real", exist_ok=True)
    os.makedirs(f"{SAVE_DIR}/fake", exist_ok=True)

    # Dataset without training augmentations (we want clean base features for evaluation)
    dataset = DeepfakeDataset(VAL_REAL_DIR, VAL_FAKE_DIR, is_train=False)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Initialize heavy models
    fmre = FrequencyMaskingModule().to(DEVICE).eval()
    fmre = load_pretrained_fmre(fmre, FIRE_WEIGHTS_PATH, DEVICE)
    vit_extractor = FrozenViTExtractor(TARGET_LAYERS).to(DEVICE).eval()

    print("--- Starting Validation Feature Extraction ---")
    
    with torch.no_grad():
        for idx, (image, label) in enumerate(tqdm(loader, desc="Validation Images")):
            image = image.to(DEVICE)
            
            # 1. Generate pseudo image
            pseudo_image = fmre(image)
            
            # 2. Extract latent features
            combined_features = vit_extractor(image, pseudo_image)
            
            # Remove batch dimension for cleaner saving
            feature_tensor = combined_features.squeeze(0).cpu() 
            
            # Determine class and save path
            is_real = label.item() == 1.0
            class_folder = "real" if is_real else "fake"
            
            save_path = f"{SAVE_DIR}/{class_folder}/feature_{idx}.pt"
            torch.save(feature_tensor, save_path)

    print(f"Extraction complete! Saved all validation features to {SAVE_DIR}/")

if __name__ == "__main__":
    run_val_extraction()