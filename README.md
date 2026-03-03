0p# Robust Signature Verification System

A deep learning–based document authentication system that verifies genuine vs forged signatures and detects adversarial manipulation attempts to ensure secure verification. 

This project implements a robust AI pipeline using transfer learning (ResNet18), adversarial attack generation, and adversarial detection to build a security-focused signature verification framework.

---

# Overview

Traditional signature verification systems can be vulnerable to sophisticated image manipulations.
This system improves reliability by:

Detecting forged signatures

Generating adversarial attacks to test model robustness

Detecting adversarial manipulation attempts

Providing a secure verification pipeline

The system performs verification in two stages:

Input Image
     ↓
Adversarial Detection
     ↓
If Safe → Signature Forgery Detection
     ↓
Final Decision

---

# Features

Genuine vs forged signature classification

Adversarial attack generation (FGSM)

Adversarial manipulation detection

Transfer learning using ResNet18

Robust verification pipeline

Real-world testing support

End-to-end AI authentication workflow

---

# Methodology

1. Signature Forgery Detection

Pretrained ResNet18 (transfer learning)

Classifies signatures as genuine or forged

Trained on real and forged signature datasets


2. Adversarial Attack Generation

FGSM (Fast Gradient Sign Method)

Generates imperceptible noise to fool the model

Demonstrates the vulnerability of baseline models


3. Adversarial Defense

Secondary model detects manipulated inputs

Prevents adversarial exploitation

Improves system robustness

---

# Model Performance

Component	Performance

Signature Forgery Detector	~84% accuracy

Adversarial Detector	~100% accuracy

---

# Tech Stack

Python

PyTorch

TorchVision

ResNet18 (Transfer Learning)

Foolbox (Adversarial Attacks)

NumPy

PIL / OpenCV

---

# Project Structure

dl project/
│
├── dataset/                  # Real and forged signature dataset
├── adv_dataset/              # Dataset for adversarial detector
│   ├── normal/
│   └── adversarial/
│
├── train.py                  # Train forgery detection model
├── train_adv_detector.py     # Train adversarial detector
├── attack.py                 # Generate adversarial images
├── predict.py                # Single image prediction
├── main.py                   # Full verification pipeline
│
├── sealguard_resnet.pth      # Trained forgery model
├── adv_detector.pth          # Trained adversarial detector

---

# Installation
Clone repository
git clone <repository-url>
cd robust-signature-verification
Install dependencies
pip install torch torchvision foolbox opencv-python matplotlib pillow

---

# Usage
Train signature forgery model
py train.py
Generate adversarial samples
py attack.py
Train an adversarial detector
py train_adv_detector.py
Run the full verification system
py main.py
Output
✔ Genuine Signature
❌ Forged Signature
⚠ Adversarial Manipulation Detected

---

# Applications

Banking signature verification

Digital document authentication

Contract validation systems

Certificate verification

Secure identity verification

Future Improvements

Web-based upload interface

Stamp and seal verification

Stronger adversarial attacks (PGD)

Explainable AI visualization

Multi-modal document verification

---

# Author
Deep Learning Project — Robust Signature Authentication with Adversarial Defence.
