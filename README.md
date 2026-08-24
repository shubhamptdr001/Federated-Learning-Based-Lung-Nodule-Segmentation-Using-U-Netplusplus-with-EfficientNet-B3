# 🧠 Federated Learning-Based Lung Nodule Segmentation using U-Net++ with EfficientNet-B3 Encoder

> A privacy-preserving federated learning pipeline for lung nodule segmentation using a **U-Net++ architecture with an EfficientNet-B3 encoder** and Dice-Focal Loss.

---

## 📘 Overview

This project presents a federated learning framework designed to perform **lung nodule segmentation** on thoracic CT scans using **U-Net++ with an EfficientNet-B3 encoder**.

The model uses U-Net++ nested skip connections together with EfficientNet-B3 for effective multi-scale feature extraction while maintaining data privacy through federated training.

### ✨ Key Contributions

* 🚀 **U-Net++ with EfficientNet-B3 encoder** for lung nodule segmentation.
* 🎯 **Dice-Focal Loss** for handling small and imbalanced lung nodules.
* 🌐 A federated setup with **5 clients and a server**.
* 📊 Comparison of **FedAvg, FedProx, FedProx++, and FedOpt (FedAdam)**.
* 🔬 Client-wise evaluation using Dice, IoU, Precision, Recall, and mIoU.

---

## 📂 Project Structure

---

## 🖼️ Dataset

### 📎 Original Dataset (LUNA16 - LIDC-IDRI)

* Dataset: `[LUNA16 Segmentation Data](https://huggingface.co/datasets/H-Huang/LUNA16_segmentation_data)`
* Preprocessed Dataset: `git clone https://huggingface.co/datasets/H-Huang/LUNA16_segmentation_data`

### 🔧 Preprocessed Dataset

Client-wise preprocessed dataset split for federated learning:

```text
client_x_images
client_x_masks
```

---

## ⚙️ Preprocessing Description

The preprocessing pipeline includes:

1. CT image preprocessing.
2. Image and mask matching.
3. Image resizing.
4. Binary mask preparation.
5. Client-wise dataset splitting.

---

## 🧠 Proposed Model: `U-Net++ + EfficientNet-B3`

### 📌 Description

A federated U-Net++ segmentation model integrating:

* ✅ **U-Net++** for nested skip connections and multi-scale feature extraction.
* ✅ **EfficientNet-B3** encoder for feature extraction.
* ✅ **Dice-Focal Loss** to handle foreground-background class imbalance.

### 🎯 Why It Excels

* Strong segmentation performance across federated clients.
* Effective feature extraction using EfficientNet-B3.
* Nested skip connections for improved spatial information.
* Designed for lung nodule segmentation under federated training.

---

## 🔬 Federated Optimizers

| Federated Optimizer  | Description                                             |
| -------------------- | ------------------------------------------------------- |
| **FedAvg**           | Baseline weighted federated averaging                   |
| **FedProx**          | FedAvg with proximal regularization                     |
| **FedProx++**        | Decaying proximal regularization with gradient clipping |
| **FedOpt (FedAdam)** | Adaptive server-side optimization                       |

---

## ⚙️ Training Configuration

| Parameter                   |           Value |
| --------------------------- | --------------: |
| Clients                     |               5 |
| Communication Rounds        |              50 |
| Local Epochs                |               2 |
| Local Learning Rate         |            1e-4 |
| FedProx μ                   |            0.01 |
| FedOpt Server Learning Rate |            1e-3 |
| Optimizer                   |           AdamW |
| Loss                        | Dice-Focal Loss |

---

## 🖥️ How to Run

> Install the requirements:

```bash
pip install -r requirements.txt
```

### 1. Run Federated Training

```bash
python train_federated.py
```

### 2. Evaluate the Models

```bash
python evaluation.py
```

---

## 🧪 Results Summary

| Federated Optimizer   | Dice Score |       mIoU |
| --------------------- | ---------: | ---------: |
| **FedAvg (Baseline)** | **0.8525** | **0.8763** |
| FedProx               |     0.8422 |     0.8683 |
| FedProx++             |     0.8449 |     0.8701 |
| FedOpt (FedAdam)      |     0.8341 |     0.8646 |

### 📊 Base Paper Comparison

| Federated Optimizer  | Base Paper Dice | Base Paper mIoU | **Our Dice** | **Our mIoU** |
| -------------------- | --------------: | --------------: | -----------: | -----------: |
| **FedAvg**           |           0.800 |           0.746 |   **0.8525** |   **0.8763** |
| **FedProx++**        |           0.812 |           0.752 |   **0.8449** |   **0.8701** |
| **FedOpt (FedAdam)** |           0.808 |           0.749 |   **0.8341** |   **0.8646** |

---

## 🔐 Why Federated Learning?

* ✅ Privacy-preserving training.
* ✅ Raw medical data remains at individual clients.
* ✅ Only model updates are exchanged.
* ✅ Supports training across distributed datasets.

---

## 💾 Best Model Checkpoints

📥 `[BEST_MODEL_CHECKPOINT](https://drive.google.com/file/d/1iJtRLT4tMqbFzevx_bYtWVSnJs1KivLH/view?usp=sharing)`

---

## 👨‍💻 Author

**Shubham Patidar**

Master of Computer Science — AI & IoT
National Institute of Technology, Patna
