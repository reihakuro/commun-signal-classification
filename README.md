# DEEP LEARNING FOR RF SIGNAL CLASSIFICATION

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/Numpy-013243?logo=numpy)
[![Kaggle](https://img.shields.io/badge/Kaggle-Profile-blue?logo=kaggle)](https://www.kaggle.com/trungkientrungkien)
![Notebook](https://img.shields.io/badge/Notebook-Jupyter-informational?logo=jupyter)
[![CI](https://github.com/reihakuro/commun-signal-classification/actions/workflows/lint.yml/badge.svg)](https://github.com/reihakuro/commun-signal-classification/actions)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

This repository focuses on Radio Frequency signal classification from spectrogram images using Convolutional Neural Networks. 
The entire training pipeline is executed on Kaggle platform.

## ✨ Features
- **Optimized Architecture**:
Uses a MobileNetV2 backbone - Inverted Residual blocks - combined with Squeeze-and-Excitation attention.

- **Lightweight Design**:
The number of trainable parameters is tightly controlled: ~96,812

- **Advanced Data Augmentation**:
MixUp (linear blending) for better generalization across spectrogram patterns
CutMix (patch substitution) for learning localized time-frequency features

- **Two-Phase Training Strategy**:
  - Phase 1: Train on 80% of the dataset using OneCycleLR scheduler
  - Phase 2: Fine-tune on 100% of the dataset using CosineAnnealingLR to improve generalization on hidden test data

## 📘 Repository Structure
```
commun-signal-classification/
├── src/                      # Source code 
│   ├── config.py             # Hyperparameter configuration
│   ├── dataset.py            # DataLoader and augmentation transforms
│   ├── model.py              # Model definition: MobileNetV2 + SE
│   ├── train.py              # Training pipeline
│   └── utils.py              # Utility functions: MixUp, CutMix, seed
├── notebooks/                # Jupyter notebooks: training and evaluation
├── test/                     # Testing scripts on test data
├── .github/workflows/ci.yml  # GitHub Actions CI configuration
├── requirements.txt          # Environment dependencies
└── README.md                 # Project documentation
```
## 📂 Dataset
### Training dataset
The model is trained on a single RF signal dataset, where raw signals are transformed into spectrogram representations for CNN-based classification.
The objective of the model is to learn and accurately classify 12 distinct RF signal categories.

- Data source: [Train Dataset](https://www.kaggle.com/datasets/huynhthethien/radarcommunsignaldata2026train)
- Representation: Time-frequency spectrograms 224x244 images
- Number of classes: 12 signals

12 signals included: 
16-QAM,
B-FM,
BPSK,
Barker,
CPFSK,
DSB-AM,
GFSK,
LFM,
PAM4,
QPSK,
Rect,
StepFM

### Testing dataset
- Data source: [Test Dataset](https://www.kaggle.com/datasets/huynhthethien/radarcommunsignaldata2026train)
- Representation: Time-frequency spectrograms 224x244 images
- Number of classes: 12 signals

## 🏗️ Model Architecture

The model utilizes a MobileNetV2-based feature extractor combined with a Squeeze-and-Excitation (SE) attention mechanism.
- Inverted Residual (MobileNetV2): utilizes depthwise separable convolutions to improve computational efficiency.
- Squeeze-and-Excitation: SE block addresses this by adaptively recalibrating the importance of each feature channel.
- ReLU6 activation function: stabilizes the training process and suitable for quantization progress.

| Layer | Block Type | Ouput Size
| :--- | :--- | :--- 
| **Input** | - | `(3, 224, 224)` 
| **Stem** | Conv2d (k=3, s=2) + BN + ReLU6 | `(16, 112, 112)` 
| **Block 1** | InvertedResidual (t=1, s=1) | `(16, 112, 112)` 
| **Block 2** | InvertedResidual (t=6, s=2) + SE | `(24, 56, 56)` 
| **Block 3** | InvertedResidual (t=6, s=1) + SE | `(24, 56, 56)` 
| **Block 4** | InvertedResidual (t=6, s=2) + SE | `(32, 28, 28)` 
| **Block 5** | InvertedResidual (t=6, s=1) + SE | `(32, 28, 28)` 
| **Block 6** | InvertedResidual (t=6, s=2) + SE | `(48, 14, 14)` 
| **Block 7** | InvertedResidual (t=4, s=2) + SE | `(64, 7, 7)` 
| **Head** | Conv2d (k=1) + BN + ReLU6 + GAP | `(128, 1, 1)` 
| **Classifier** | Dropout (p=0.35) -> Linear | `(num_classes)` 

## ⛏️ Training
- Optimizer: AdamW
- Scheduler: OneCycleLR at phase 1, CosineAnnealingLR at phase 2
- Loss: CrossEntropyLoss and Label Smoothing
- Augmentation: MixUp, CutMix, RandomHorizontalFlip, RandomErasing

## 🏅 Result
### Overall

Best Phase-1 Validation Accuracy: 0.9012

Overall Accuracy: 0.9258
```
Class          Precision      Recall    F1-score   Support
----------------------------------------------------------
16-QAM            0.7776      0.8221      0.7992      5150
B-FM              0.9919      0.9834      0.9876      5111
BPSK              0.9016      0.7977      0.8464      5110
Barker            0.9998      0.9980      0.9989      5125
CPFSK             0.9865      0.9751      0.9808      5190
DSB-AM            0.8047      0.9937      0.8893      5091
GFSK              0.9941      0.9880      0.9910      5148
LFM               1.0000      0.9998      0.9999      5089
PAM4              0.8517      0.8370      0.8443      5098
QPSK              0.8632      0.7495      0.8023      5145
Rect              0.9702      0.9953      0.9826      5095
StepFM            0.9950      0.9711      0.9829      5088
```
### Confusion Matrix

```
              Predicted →
True ↓     16QAM  B-FM  BPSK  Barker  CPFSK  DSB-AM  GFSK  LFM  PAM4  QPSK  Rect  StepFM
----------------------------------------------------------------------------------------
16-QAM      4007     1   151      0     16     322     13    0   182   458     0      0
B-FM           7  5019    12      0      8      36      3    0    14    12     0      0
BPSK         116     3  4206      0     11     325     11    0   281   157     0      0
Barker         0     0     0   5114      0       0      0    0     0     0     7      4
CPFSK         23     1    17      0   5064      34      0    0    11    40     0      0
DSB-AM         3     0     9      0      0    5062      3    0     6     8     0      0
GFSK           3     1     6      0      0      37   5092    0     3     6     0      0
LFM            0     0     0      0      0       0      0  5088    0     0     0      1
PAM4         147     6   317      0      9     281     10    0  4195   133     0      0
QPSK         438     4   203      0     21     320     11    0   131  4017     0      0
Rect           0     0     0      0      0       0      0    0     0     0  5072     23
StepFM         0     0     0      0      0       0      0    0     0   220     0   4868
```
### Evaluation on Test Dataset
```
Total test samples: 307200

==============================
EVALUATING MODEL
==============================
Model: 04_DeepLearningProject_TrainedModel.pt

==============================
RESULT
==============================
Group ID : 04
Accuracy : 0.8979
Correct  : 275831/307200
```
## ⭐ Acknowledgments
The authors would like to express their sincere gratitude to Dr. Huynh The Thien 
for providing the dataset and for his valuable guidance throughout the project.
We also acknowledge the use of PyTorch and Kaggle as the primary platforms for development and experimentation.

#
<sup>Pham Trung Kien - reihakuro</sup>

<sup>and</sup>

<sup>Group04-MLAI03-2026, UTE</sup>

<sup>Vu Phan Thanh Dat</sup>

<sup>Chong Dinh Khang</sup>

<sup>Hoang Bao Phuc</sup>

<sup>Le Thanh Tu</sup>
