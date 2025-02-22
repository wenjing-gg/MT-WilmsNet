# MT-WilmsNet: A Multi-Level Transformer Fusion Network for Wilms’ Tumor Segmentation and Metastasis Prediction

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Future Work](#future-work)
- [Citation](#citation)

## Overview
A multi-task framework combining tumor segmentation and metastasis prediction using CT imaging data. Designed to reduce dependency on PET-CT scans while maintaining diagnostic accuracy for Wilms’ Tumor (WT).

![Model Architecture](path/to/architecture_diagram.png) <!-- Add your diagram file -->

## Requirements
- Python ≥ 3.8
- PyTorch ≥ 2.0.0
- Transformers ≥ 4.30.0
- SimpleITK ≥ 2.2.0 (for medical image processing)
- NVIDIA CUDA ≥ 11.7
- 16GB+ VRAM GPU recommended

Install dependencies:
```bash
pip install torch torchvision transformers SimpleITK
