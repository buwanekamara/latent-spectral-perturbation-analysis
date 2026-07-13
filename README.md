# Latent Spectral Perturbation Analysis (LSPA) for Deepfake Detection

An academic research framework for AI-generated image detection. This project implements a decoupled, high-efficiency pipeline that utilizes frequency-domain masking as a latent space stressor, measuring representation collapse across intermediate Vision Transformer (ViT) layers through a dynamic gating network.

---

## 1. Academic Novelty & Scope

The LSPA framework sits at the intersection of frequency analysis (FIRE) and intermediate latent representation (MoLD), introducing three highly defensible novelties:

* **Cross-Domain Stress Testing:** Rather than analyzing static pixels or static frequencies, LSPA uses frequency perturbation as a "stressor" and measures how a Vision Transformer *reacts* to that stress.
* **Delta-Context Feature Concatenation:** Avoiding the information loss of traditional Cosine Similarity, the architecture calculates the absolute dimensional difference (Delta) between pristine and stressed states, binding it directly to the original context to preserve a full 2048-dimensional topological map of the disruption.
* **Dynamic Mid-Layer Gating:** Capitalizing on the vulnerability of specific mid-layers to frequency manipulation, a custom Multi-Layer Perceptron (MLP) dynamically learns to "vote" on which layers suffered the worst representation collapse, discarding the traditional reliance on the ViT's final semantic layer.

---

## 2. Core Architecture

The pipeline operates across three decoupled modules designed for low-compute environments without sacrificing feature depth.

### Module 1: The Stressor (Frequency Masking)
* Utilizes a frozen ESPCN-backed network (derived from FIRE).
* Strips vulnerable mid-band frequencies from the input image via FFT analysis, generating a frequency-stressed pseudo-image.

### Module 2: The Observer (Latent Feature Extractor)
* Feeds both the pristine and stressed images through a frozen `openai/clip-vit-large-patch14`.
* Extracts the `[CLS]` token hidden states explicitly from mid-layers **11, 13, 15, and 17**. 
* *Why these layers?* According to MoLD, a ViT operates as a hierarchy. While late layers encode high-level semantics (which modern AI fakes perfectly), mid-layers encode texture and structural integrity. Stripping mid-frequencies triggers a massive, measurable mathematical collapse exclusively in these mid-layers.
* Computes the absolute difference:
  $$\Delta=\vert{}z_{\text{orig}}-z_{\text{stressed}}\vert{}$$

### Module 3: The Judge (Dynamic Gating Network)
* A custom, trainable 3-layer MLP featuring aggressive dropout ($p=0.3$) and dimensionality reduction.
* Independently scores the anomaly level of each layer's 2048-dimensional feature block, fusing them to output a final continuous binary classification ($1.0$ = Real, $0.0$ = Fake).

---

## 3. Setup & Installation

This project relies on the ultra-fast `uv` package manager and requires a strictly controlled Python environment to ensure CUDA C++ extensions map correctly to local hardware.

**1. Create a stable isolated environment (Requires Python 3.12 or 3.13):**
```bash
uv venv --python 3.12


---
## 4.folder structure

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