# DDPM-Based Sprite Generator 🎮✨ (Context-Aware)

This project implements a **Denoising Diffusion Probabilistic Model (DDPM)** to generate **16x16 pixel sprites**, with and without context conditioning.

The model is trained to **predict noise** in a diffusion process, enabling both **unconditional** and **class-conditional** generation of sprites.

---

## 🛠️ Project Structure

- `sprites.npy`: Sprite image dataset (16x16 RGB images)
- `sprite_labels.npy`: Context labels (one-hot encoded classes)
- `ContextUNet`: UNet-inspired architecture with context injection
- `DDPMSampler`: Standard diffusion sampling (reverse process)
- `train.py`: Full training and sampling pipeline (this file)

---

## 🚀 How It Works

- **Training Phase**:
  - Add controlled Gaussian noise to clean images at random diffusion steps.
  - Train the network to predict the added noise, optionally conditioned on class labels.

- **Sampling Phase (DDPM)**:
  - Start from pure random noise.
  - Iteratively denoise step-by-step using the learned noise predictor.
  - Output: A fresh, realistic-looking sprite image.

---

## 🧩 Model Details

- **Input**: 16×16 RGB sprite images
- **Conditioning**: One-hot class labels for conditional generation
- **Architecture**:
  - 2 convolutional layers
  - Linear context embedding and fusion
  - Final 1x1 convolution for RGB output

- **Sampling**: Standard DDPM step-by-step denoising



---

## 🛠️ Requirements

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib

Install dependencies:
```bash
pip install torch numpy matplotlib
