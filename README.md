# Super-diffractive-limit-reconstruction-diffusion-model
---

## 📘 Overview

This project proposes a hybrid Denoising Diffusion Probabilistic Model (DDPM) that surpasses the diffraction limit for sub-diffraction target recovery.  
The model integrates spatial and channel attention mechanisms, including **SCAU** (Selective Channel Attention Unit) and **SGFE** (Spatial Gated Feature Enhancement), to recover high-frequency details that conventional methods fail to reconstruct.

---

## 🏗️ Project Structure
main/
├── models/ # Core model implementations (SCAU, SGFE, hybrid DDPM, etc.)
├── train.py # Training script (to be added)
├── inference.py # Inference/testing script (to be added)
└── README.md


---

## 📂 Dataset

The experiments are conducted on:
1. A **real-world dataset**, and  
2. A **synthetic dataset** simulated from three public benchmarks.

👉 **Dataset Download Link (to be inserted later):**  
`[Insert dataset link here once available]`

Please organize the dataset as follows:
datasets/
├── real/
│ ├── train/
│ ├── val/
│ └── test/
└── synthetic/
├── train/
├── val/
└── test/


---

## ⚙️ Environment Setup

This implementation is based on **PyTorch** and tested on:

- **OS:** Ubuntu 18.04  
- **GPU:** NVIDIA A40  
- **CUDA:** 11.3  
- **cuDNN:** 8  
- **Python:** 3.8+


