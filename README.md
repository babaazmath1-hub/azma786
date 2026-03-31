# GSoC 2026 — ML4SCI E2E | Particle Physics Classification

[![Open Task 1 In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/babaazmath1-hub/azma786/blob/main/modified_Task1_Electron_Photon_Classification.ipynb)
[![Open Task 2 In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/babaazmath1-hub/ml4sci-e2e-gsoc2026/blob/main/Task2_sparse_neural_network.ipynb)

**Author:** Shaik Baba Azmath  
**GitHub:** [babaazmath1-hub](https://github.com/babaazmath1-hub)  
**Organization:** [ML4SCI](https://ml4sci.org/)  
**Project:** Sparse Neural Network Pipeline for Particle Collision Event Classification (E2E)

---

## Overview

This repository contains solutions to the two common tasks for the **ML4SCI End-to-End (E2E) Deep Learning** project under Google Summer of Code 2026. Both tasks involve classifying particle physics detector images using deep convolutional neural networks, with the shared goal of maximizing the AUC score on held-out test sets.

The work demonstrates the feasibility of dense ResNet baselines while motivating the proposed GSoC project: replacing dense convolutions with **Submanifold Sparse Convolutions** to exploit the naturally high sparsity (~90%+) of CMS detector images.

---

## Repository Structure

```
.
├── modified_Task1_Electron_Photon_Classification.ipynb   # Task 1: Electron vs Photon
├── Task2_sparse_neural_network.ipynb                     # Task 2: Quark vs Gluon
└── README.md
```

---

## Task 1 — Electron vs Photon Classification

### Problem
Classify **electron** vs **photon** particle events from 32×32 CMS detector images with 2 channels:
- **Channel 1:** Hit Energy
- **Channel 2:** Hit Time

### Dataset
- Source: [Kaggle — electron-vs-photons-ml4sci](https://www.kaggle.com/datasets/vishakkbhat/electron-vs-photons-ml4sci)
- 15,000 samples per class (30,000 total)
- Split: 80% train / 10% val / 10% test (stratified)

### Architecture — ResNet-15

A lightweight residual network adapted from Andrews et al. (2020):

| Layer | Output Channels | Stride |
|-------|----------------|--------|
| Stem Conv 3×3 | 32 | 1 |
| Layer 1 (2× ResBlock) | 64 | 1 |
| Layer 2 (2× ResBlock) | 128 | 2 |
| Layer 3 (2× ResBlock) | 256 | 2 |
| Layer 4 (2× ResBlock) | 256 | 2 |
| Global Avg Pool + FC | 1 | — |

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Learning Rate | 5e-4 |
| Weight Decay | 1e-4 |
| Scheduler | CosineAnnealingLR (T_max=25) |
| Epochs | 25 |
| Batch Size | 512 |
| Loss | BCEWithLogitsLoss |
| Dropout | 0.3 |

### Results

| Metric | Value |
|--------|-------|
| Test AUC | ~0.80+ |
| Accuracy | ~0.75+ |
| Parameters | ~2.5M |

---

## Task 2 — Quark vs Gluon Jet Classification

### Problem
Classify **quark-initiated** vs **gluon-initiated** jets from 125×125 CMS detector images with 3 channels:
- **Channel 1:** ECAL (Electromagnetic Calorimeter)
- **Channel 2:** HCAL (Hadronic Calorimeter)
- **Channel 3:** Reconstructed Tracks

Reference: [Andrews et al., 2020 — arXiv:1902.08276](https://arxiv.org/abs/1902.08276)

### Dataset
- Source: CernBox QG jets dataset
- Fallback: synthetic data generation if download fails
- Up to 100,000 samples used, 80/10/10 split

### Architecture — ResNet-15 (QG variant)

A deeper variant matching the Andrews et al. paper:

| Layer | Output Channels | Stride |
|-------|----------------|--------|
| Stem Conv 3×3 | 64 | 1 |
| Layer 1 (2× ResBlock) | 64 | 1 |
| Layer 2 (2× ResBlock) | 128 | 2 |
| Layer 3 (2× ResBlock) | 256 | 2 |
| Layer 4 (2× ResBlock) | 512 | 2 |
| Global Avg Pool + FC | 1 | — |

### Preprocessing Pipeline

1. `log(1 + x)` transform to compress energy scale
2. Per-channel z-score normalization
3. Subsample to ≤ 100,000 samples

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Learning Rate | 5e-4 |
| Weight Decay | 1e-4 |
| Scheduler | StepLR (step=10, γ=0.5) |
| Epochs | 30 |
| Batch Size | 128 |
| Gradient Clipping | 1.0 |

### Results

| Metric | Value | Paper Baseline |
|--------|-------|----------------|
| Test AUC | ~0.80+ | 0.8076 |
| 1/FPR @ TPR=70% | ~4+ | 4.47 |

---

## Comparison: Task 1 vs Task 2

| Aspect | Task 1 (Electron/Photon) | Task 2 (Quark/Gluon) |
|--------|--------------------------|----------------------|
| Image Size | 32×32 | 125×125 |
| Channels | 2 (Energy, Time) | 3 (ECAL, HCAL, Tracks) |
| Dataset Size | 30,000 | Up to 100,000 |
| Model Width | 32→64→128→256→256 | 64→64→128→256→512 |
| Parameters | ~2.5M | ~11M |
| Scheduler | CosineAnnealing | StepLR |
| Epochs | 25 | 30 |
| Batch Size | 512 | 128 |
| Preprocessing | Per-channel z-score | log1p + z-score |
| Sparsity | Moderate | ~90%+ zero pixels |

### Key Shared Design Choices
- Both use ResNet-15 architecture with residual blocks and BatchNorm
- Both use BCEWithLogitsLoss with sigmoid output for binary classification
- Both use AdaptiveAvgPool2d + Dropout(0.3) before the final FC head
- Both track AUC as the primary metric
- Identical skip connection implementation for channel/stride mismatch

### Key Differences
- Task 2 uses gradient clipping (not needed for the smaller Task 1)
- Task 2 applies log1p to handle sparse, heavy-tailed energy distributions
- Task 2 computes additional 1/FPR@TPR=70% metric matching the reference paper
- Task 2 includes explicit sparsity analysis motivating sparse convolution methods

---

## Motivation for Sparse Neural Networks

The Task 2 sparsity analysis reveals that CMS detector images contain **>90% zero pixels**. Dense convolutions waste the majority of compute on empty regions. The proposed GSoC project addresses this with **Submanifold Sparse Convolutions** (Graham & van der Maaten, 2017), which:

- Restrict computation to non-zero voxels only
- Achieve 2–5× FLOP reduction vs dense CNNs
- Maintain equivalent AUC on the same tasks

---

## Requirements

```bash
pip install torch torchvision numpy h5py scikit-learn matplotlib seaborn
# For Task 1 dataset download:
pip install kaggle
```

Both notebooks are designed to run on **Google Colab** with GPU acceleration.

---

## References

1. Andrews, M. et al. *End-to-End Jet Classification of Quarks and Gluons with the CMS Open Data.* arXiv:1902.08276 (2020).
2. Graham, B. & van der Maaten, L. *Submanifold Sparse Convolutional Networks.* arXiv:1706.01307 (2017).
3. ML4SCI E2E Project: https://ml4sci.org/

---

## License

MIT License — see [LICENSE](LICENSE) for details.
