# Edge-Based Super-Resolution Deployment for Low-Resolution Agricultural Imaging

This repository provides the implementation for deploying deep learning–based super-resolution (SR) models on an edge computing platform. The work focuses on analyzing **quality–efficiency trade-offs** in real-time SR deployment under resource-constrained edge environments, using low-resolution agricultural images acquired from an ESP32-CAM.

This code accompanies the research paper:

**Quality–Efficiency Trade-Offs in Real-Time Super-Resolution Deployment on Edge Vision Systems**

---

## 📌 Overview

Low-cost embedded cameras such as **ESP32-CAM** are widely adopted in smart agriculture due to their affordability and low power consumption. However, the limited image resolution produced by such devices poses challenges for downstream vision-based analysis.

This project evaluates the deployment of three deep learning–based super-resolution models on an edge device to enhance image resolution by **2×**, from **800×600 (SVGA)** to **1600×1200 (UXGA)**, using real-world lettuce plant images.

The evaluation emphasizes not only reconstruction quality, but also **deployment-aware performance**, including processing time, computational load, and power consumption.

---

## 🧠 Super-Resolution Models

The following state-of-the-art SR models are evaluated:

- **EDSR** – Residual CNN-based super-resolution  
- **Real-ESRGAN** – GAN-based perceptual super-resolution  
- **ESPCN** – Lightweight sub-pixel convolution network  

The implementations are adapted from publicly available repositories to support edge deployment and real-time inference.

---
## Dataset Availability

The dataset used in this project consists of real images captured using an ESP32-CAM in an agricultural monitoring setup. 
Due to data usage considerations, the dataset is not publicly released. 
However, it can be made available for academic and research purposes upon reasonable request by contacting the corresponding author.

---

## ⚙️ System Configuration

- **Edge Device**: NVIDIA Jetson Orin Nano  
- **Camera**: ESP32-CAM  
- **Framework**: PyTorch  
- **Execution Modes**:
  - GPU (CUDA acceleration)
  - CPU (baseline comparison)

---

## 🔁 Reproducibility and Online Resources

This work relies on publicly available implementations of SR models:

Real-ESRGAN
https://github.com/xinntao/Real-ESRGAN

EDSR (PyTorch)
https://github.com/sanghyun-son/EDSR-PyTorch

ESPCN (PyTorch)
https://github.com/Lornatang/ESPCN-PyTorch

All experiments were conducted using real low-resolution images captured with ESP32-CAM. Training configurations, inference procedures, and deployment settings are described in detail in the accompanying paper.
For
