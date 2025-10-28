# Super-diffractive-limit-reconstruction-diffusion-model
---

## 📘 Overview

This project proposes a hybrid Denoising Diffusion Probabilistic Model (DDPM) that surpasses the diffraction limit for sub-diffraction target recovery.  
The model integrates spatial and channel attention mechanisms, including **SCAU** and **SGFE** , to recover high-frequency details that conventional methods fail to reconstruct.

---

## 🏗️ Project Structure
 models/ # Core model implementations (SCAU, SGFE, hybrid DDPM, etc.)

---

## 📂 Dataset

The experiments are conducted on:
1. A **real-world dataset**, and  
2. A **synthetic dataset** simulated from three public benchmarks.
Due to institutional restrictions, the full dataset cannot be publicly released. However, we have provided representative samples and key code descriptions at
👉 **Dataset Download Link:**  
`[https://pan.baidu.com/s/1UIsFMfnWuA010w94MD9xDA?pwd=wxvi 提取码: wxvi]`

---

## ⚙️ Environment Setup

This implementation is based on **PyTorch** and tested on:

- **OS:** Ubuntu 18.04  
- **GPU:** NVIDIA A40  
- **CUDA:** 11.3  
- **cuDNN:** 8  
- **Python:** 3.8+


