Berikut adalah **kode `README.md` siap pakai** untuk repositori GitHub Anda. Anda bisa **langsung menyalin seluruh isi di bawah ini** ke file `README.md` tanpa perlu modifikasi tambahan.

````markdown
# Fine-Tuning TinyBERT for Multi-Label Emotion Classification (GoEmotions)

## 📥 Pre-trained Model Availability

> **Note:** The fine-tuned model generated in this project can be saved locally after training. Due to repository size limitations, model weights are **not directly included** in this repository. Inference can be performed immediately after training or by loading the saved model directory.

---

## 📔 Table of Contents

- [Purpose](#-purpose)
- [Project Overview](#-project-overview)
- [Dataset Preparation](#-dataset-preparation)
- [Tokenization and Labeling](#-tokenization-and-labeling)
- [Model Configuration](#-model-configuration)
- [Training Details](#-training-details)
- [Inference and Usage](#-inference-and-usage)
- [Results and Observations](#-results-and-observations)
- [Repository Structure](#-repository-structure)
- [Installation & Setup](#-installation--setup)
- [Conclusion](#-conclusion)

---

## 🎯 Purpose

This project focuses on fine-tuning the **TinyBERT (4L-312D)** Encoder-only Transformer for **multi-label text classification** using the **GoEmotions** dataset. The main objective is to build an efficient and lightweight emotion classification model capable of detecting **28 distinct human emotions** from text, optimized for environments with limited computational resources such as Google Colab Free.

---

## 🔍 Project Overview

### Multi-Label Classification

Unlike traditional multi-class classification (one label per sample), this task allows **multiple emotions to be active simultaneously** (e.g., *Joy* and *Optimism*). Therefore, the model uses **Sigmoid activation** instead of Softmax to independently estimate probabilities for each label.

### TinyBERT Model

This project uses **Huawei Noah’s TinyBERT** (`huawei-noah/TinyBERT_General_4L_312D`), a compressed BERT model obtained via knowledge distillation. Compared to BERT-Base (12 layers), TinyBERT uses only 4 layers while maintaining strong semantic understanding, making it suitable for fast experimentation and edge deployment.

---

## 📊 Dataset Preparation

### Dataset Source

The **GoEmotions dataset (Simplified configuration)** is used. It consists of Reddit comments annotated with **27 emotion labels plus a Neutral class**, resulting in 28 total labels.

### Preprocessing Strategy

To support multi-label classification:

1. **Multi-Hot Encoding**  
   Each sample’s labels are converted into a binary vector of length 28.
2. **Float Casting**  
   Labels are cast to `float32` to ensure compatibility with `BCEWithLogitsLoss`.

```python
# Example label transformation
Original: [3, 27]  # Indices for 'Anger' and 'Neutral'
Transformed: [0.0, 0.0, 0.0, 1.0, ..., 1.0]  # Float multi-hot vector
````

---

## 🧠 Tokenization and Labeling

* **Tokenizer:** `AutoTokenizer` from `huawei-noah/TinyBERT_General_4L_312D`
* **Sequence Length:** Padded and truncated to a maximum of **128 tokens**
* **Custom Data Collator:**
  A custom `data_collator` is used to force label tensors into `torch.float32`, preventing runtime errors such as:

```
RuntimeError: result type Float can't be cast to the desired output type Long
```

---

## ⚙️ Model Configuration

The model is initialized using `AutoModelForSequenceClassification` with:

* **Problem Type:** `multi_label_classification`
* **Number of Labels:** 28
* **Loss Function:** `BCEWithLogitsLoss` (automatically selected by Hugging Face Trainer)
* **Precision:** Mixed precision training (`fp16`)
* **Fine-Tuning Strategy:** Full encoder fine-tuning

---

## 🏋️ Training Details

* **Platform:** Google Colab
* **GPU:** Tesla T4
* **Batch Size:** 16 (train & evaluation)
* **Epochs:** 4
* **Learning Rate:** 2e-5
* **Evaluation Metric:** Micro-averaged F1 Score (robust to class imbalance)

---

## 🚀 Inference and Usage

Inference requires post-processing of model outputs:

1. **Sigmoid Activation**
   Converts logits into probabilities.
2. **Thresholding**
   Determines which emotions are active.

* Default threshold: `0.5`
* Recommended threshold: **0.25 – 0.30**
  (Improves recall for subtle emotions in small models)

```python
# Inference example
probs = torch.sigmoid(logits).cpu().numpy()[0]
predictions = (probs >= 0.25).astype(int)
```

---

## 📈 Results and Observations

* **Efficiency:** Training time is significantly faster compared to BERT-Base.
* **Performance:** The model effectively captures dominant emotions such as *Admiration* in achievement-related text.
* **Limitations:** Smaller model capacity can distribute confidence across semantically similar emotions (e.g., *Joy*, *Love*, *Admiration*). Threshold tuning is essential for optimal results.

---

## 📁 Repository Structure

```text
finetuning-tinybert-goemotions/
│
├── notebooks/
│   └── task1_tinybert_goemotions.ipynb
├── reports/
│   └── task1_encoder_classification_report.md
├── README.md
└── requirements.txt
```

---

## 🛠️ Installation & Setup

### Requirements

* Python 3.8+
* PyTorch
* Hugging Face `transformers` and `datasets`
* Scikit-learn
* Accelerate

### Installation

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## ✅ Conclusion

This project demonstrates that **TinyBERT** can be effectively fine-tuned for **multi-label emotion classification**. Through careful label preprocessing, custom data collation, and threshold adjustment during inference, meaningful emotional insights can be extracted even with a highly compressed Transformer model suitable for resource-constrained environments.

```

