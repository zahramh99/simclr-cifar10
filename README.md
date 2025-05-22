# SimCLR Self-Supervised Learning on CIFAR-10 🌅

This project implements **SimCLR (A Simple Framework for Contrastive Learning of Visual Representations)** to learn meaningful image features from unlabeled CIFAR-10 data, then evaluates the learned representations through linear probing.

## Key Features ✨
- 🧠 Pure PyTorch implementation of SimCLR
- 🖼️ Works with CIFAR-10 out-of-the-box (no labels needed for pretraining)
- 📊 Tracks training metrics (loss curves, feature visualizations)
- ⚡ GPU-accelerated training
- 🔄 Easy to adapt to custom datasets

## Prerequisites
- Python 3.8+
- NVIDIA GPU (recommended) with CUDA 11.3+
- PyTorch 2.0+

## Installation 🛠️
```bash
# Clone repository
git clone https://github.com/zahramh99/simclr-cifar10.git
cd simclr-cifar10

# Install dependencies
pip install -r requirements.txt
