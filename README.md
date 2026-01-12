Berikut adalah file **README.md** yang telah diperbarui dan disusun rapi mengikuti struktur profesional yang Anda inginkan. File ini mencakup semua detail teknis perbaikan yang telah kita lakukan (seperti *Custom Data Collator* dan *Thresholding*).

Silakan copy-paste kode di bawah ini ke dalam file `README.md` di repository Anda.

```markdown
# Fine-Tuning TinyBERT for Multi-Label Emotion Classification (GoEmotions)

## 📥 Pre-trained Model Availability

The fine-tuned model generated in this project can be saved locally after training.  
Due to repository size limitations, model weights are not directly included in this repository.  
Inference can be performed immediately after training or by loading the saved model directory.

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

This project focuses on fine-tuning the **TinyBERT (4L-312D) Encoder-only Transformer** for **multi-label text classification** using the **GoEmotions dataset**. The main objective is to create an efficient, lightweight model capable of detecting 28 distinct human emotions from text, designed to run effectively on constrained resources like Google Colab Free.

---

## 🔍 Project Overview

### Multi-Label Classification

Unlike standard multi-class classification where a text belongs to only one category, this task requires the model to predict multiple active emotions simultaneously (e.g., a sentence can be both "Joy" and "Optimism"). This requires using **Sigmoid activation** rather than Softmax.

### TinyBERT Model

**Huawei Noah's TinyBERT** (`huawei-noah/TinyBERT_General_4L_312D`) is a compressed version of BERT produced via knowledge distillation. It retains the semantic understanding of BERT-Base while being significantly faster (approx. 4 layers vs 12 layers) and smaller, making it ideal for edge deployment and rapid experimentation.

---

## 📊 Dataset Preparation

### Dataset Source

The **GoEmotions** dataset (Simplified configuration) is used, consisting of Reddit comments labeled with 27 emotion categories plus "Neutral".

### Preprocessing Strategy

To handle the multi-label nature of the data:
- Labels are converted into a **Multi-Hot Encoding** format (a vector of 0s and 1s).
- Labels are strictly cast to `float32` to ensure compatibility with the `BCEWithLogitsLoss` function used during training.

```python
# Example Label Transformation
Original: [3, 27]  # (Indices for 'Anger' and 'Neutral')
Transformed: [0.0, 0.0, 0.0, 1.0, ..., 1.0]  # (Float vector)

```

---

## 🧠 Tokenization and Labeling

* **Tokenizer:** Uses the `AutoTokenizer` from `huawei-noah/TinyBERT_General_4L_312D`.
* **Truncation:** All sequences are padded and truncated to a maximum length of **128 tokens**.
* **Custom Collator:** A custom `data_collator` is implemented to intercept batches before training. It explicitly forces label tensors into `torch.float32` format. This is critical to prevent the common PyTorch error: `RuntimeError: result type Float can't be cast to the desired output type Long`.

---

## ⚙️ Model Configuration

The model is initialized using `AutoModelForSequenceClassification` with the following specifics:

* **Problem Type:** `multi_label_classification`.
* **Number of Labels:** 28.
* **Optimization:** Mixed Precision (`fp16`) is enabled to accelerate training on T4 GPUs.
* **Loss Function:** `BCEWithLogitsLoss` (Binary Cross Entropy with Logits) is used automatically by the Trainer API for multi-label tasks.

---

## 🏋️ Training Details

* **Hardware:** Google Colab (Tesla T4 GPU).
* **Batch Size:** 16 (Train & Eval).
* **Epochs:** 4 (Adjustable depending on convergence).
* **Learning Rate:** 2e-5.
* **Metric:** F1-Micro Score (suited for class imbalance).
* **Strategy:** Full fine-tuning of the encoder layers.

---

## 🚀 Inference and Usage

Inference requires a specific post-processing step to interpret the logits:

1. **Sigmoid Activation:** Converts raw logits into probabilities (0.0 to 1.0).
2. **Thresholding:** A threshold is applied to determine active labels.
* *Default:* 0.5
* *Recommended:* **0.25 - 0.30** (Given the model size, a lower threshold improves recall for subtle emotions).



```python
# Inference snippet
probs = torch.sigmoid(logits).cpu().numpy()[0]
predictions = (probs >= 0.25).astype(int)

```

---

## 📈 Results and Observations

* **Efficiency:** TinyBERT completes training significantly faster than BERT-Base, making it suitable for rapid iteration.
* **Performance:** The model effectively captures dominant sentiments (e.g., "Admiration" in achievement-related texts).
* **Analysis:** Due to the model's small size (distilled), confidence scores for nuanced emotions (like "Joy" vs "Love") may be diluted. Adjusting the decision threshold is necessary to balance Precision and Recall.

---

## 📁 Repository Structure

```
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
* Hugging Face Transformers & Datasets
* Scikit-learn & Accelerate

Dependencies can be installed using:

```bash
pip install -r requirements.txt

```

---

## ✅ Conclusion

This project demonstrates that **TinyBERT** can effectively serve as a multi-label emotion classifier. By implementing robust data processing (custom collators) and adjusting inference thresholds, we can extract meaningful emotional insights from text even with a highly compressed model architecture suitable for resource-constrained environments.

```

```
