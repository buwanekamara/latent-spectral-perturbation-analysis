# Latent Spectral Perturbation Analysis (LSPA)

## Overview
LSPA is an AI-generated image detection pipeline designed to identify synthetic imagery by combining frequency-domain stress testing with latent representation analysis. 

Instead of relying on computationally expensive pixel-level reconstructions, this architecture surgically removes mid-frequency bands (which are notoriously fragile in AI-generated images) and measures the resulting "representation collapse" within the latent space of a frozen Vision Transformer (ViT).

## Architecture
The pipeline consists of three sequential modules optimized for low-compute environments (e.g., Google Colab T4):

1. **Module 1: Adaptive Spectral Perturbation (The Stressor)**
   * Utilizes a frozen ESPCN network to analyze the FFT magnitude spectrum of an input image.
   * Isolates and removes mid-frequency bands to generate a "stressed" pseudo-image.
2. **Module 2: Hierarchical Latent Probing (The Observer)**
   * Passes both the original and stressed images through a frozen `openai/clip-vit-large-patch14`.
   * Extracts `[CLS]` tokens from intermediate layers (11, 13, 15, 17) and calculates the absolute difference (delta) combined with the original context.
3. **Module 3: Dynamic Gated Classification (The Judge)**
   * A custom, trainable 3-layer MLP featuring aggressive dropout (p=0.3) and dimensionality reduction (2048 -> 512).
   * Scores the anomaly level of each layer and fuses them to output a final binary classification (1.0 = Real, 0.0 = Fake).

## Project Structure
```text
lspa_project/
├── data/                   
│   ├── train/
│   │   ├── real/           # Place authentic training images here
│   │   └── fake/           # Place AI-generated training images here
│   └── val/                # Validation data (Optional)
├── models/                 
│   ├── espcn.py            # Frequency masking network
│   ├── vit_extractor.py    # CLIP-ViT feature extraction
│   └── classifier.py       # Trainable Gated MLP
├── utils/                  
│   └── dataset.py          # PyTorch dataset with zero-cost augmentations
├── train.py                # Main training loop
├── requirements.txt        
└── README.md