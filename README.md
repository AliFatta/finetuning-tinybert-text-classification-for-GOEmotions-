# Fine-Tuning TinyBERT for Multi-Label Emotion Classification (GoEmotions)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/docs/transformers)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

A lightweight emotion classification system that fine-tunes **TinyBERT** on the **GoEmotions** dataset to detect 28 distinct human emotions from text. Optimized for resource-constrained environments like Google Colab Free.

---

## 📥 Pre-trained Model Availability

The fine-tuned model generated in this project can be saved locally after training.

> **Note:** Due to repository size limitations, model weights are not directly included in this repository. Inference can be performed immediately after training or by loading the saved model directory.

---

## 📔 Table of Contents

- [Purpose](#-purpose)
- [Project Overview](#-project-overview)
- [Dataset Preparation](#-dataset-preparation)
- [Tokenization and Labeling](#-tokenization-and-labeling)
- [Model Configuration](#️-model-configuration)
- [Training Details](#️-training-details)
- [Inference and Usage](#-inference-and-usage)
- [Results and Observations](#-results-and-observations)
- [Repository Structure](#-repository-structure)
- [Installation & Setup](#️-installation--setup)
- [Conclusion](#-conclusion)

---

## 🎯 Purpose

This project focuses on fine-tuning the **TinyBERT (4L-312D) Encoder-only Transformer** for **multi-label text classification** using the **GoEmotions dataset**. The main objective is to create an efficient, lightweight model capable of detecting 28 distinct human emotions from text, running effectively on constrained resources like Google Colab Free.

---

## 🔍 Project Overview

### Multi-Label Classification

Unlike standard multi-class classification where a text belongs to only one category, this task requires the model to predict multiple active emotions simultaneously (e.g., a sentence can be both "Joy" and "Optimism"). This requires using **Sigmoid activation** rather than Softmax.

### TinyBERT Model

**Huawei Noah's TinyBERT** (`huawei-noah/TinyBERT_General_4L_312D`) is a compressed version of BERT produced via knowledge distillation. It retains the semantic understanding of BERT-Base while being significantly faster (approx. 4 layers vs 12 layers) and smaller, making it ideal for edge deployment and rapid experimentation.

**Key Advantages:**
- ⚡ 4x faster inference than BERT-Base
- 💾 Significantly smaller model size
- 🎯 Maintains strong semantic understanding
- 🚀 Perfect for edge deployment

---

## 📊 Dataset Preparation

### Dataset Source

The **GoEmotions** dataset (Simplified configuration) is used, consisting of Reddit comments labeled with 27 emotion categories plus "Neutral".

**Emotion Categories:**
```
admiration, amusement, anger, annoyance, approval, caring, confusion, 
curiosity, desire, disappointment, disapproval, disgust, embarrassment, 
excitement, fear, gratitude, grief, joy, love, nervousness, optimism, 
pride, realization, relief, remorse, sadness, surprise, neutral
```

### Preprocessing Strategy

To handle the multi-label nature of the data:

- Labels are converted into a **Multi-Hot Encoding** format (a vector of 0s and 1s)
- Labels are strictly cast to `float32` to ensure compatibility with the `BCEWithLogitsLoss` function used during training

```python
# Example Label Transformation
Original: [3, 27]  # (Indices for 'Anger' and 'Neutral')
Transformed: [0.0, 0.0, 0.0, 1.0, ..., 1.0]  # (Float vector of length 28)
```

---

## 🧠 Tokenization and Labeling

- **Tokenizer:** Uses the `AutoTokenizer` from `huawei-noah/TinyBERT_General_4L_312D`
- **Truncation:** All sequences are padded and truncated to a maximum length of **128 tokens**
- **Custom Collator:** A custom `data_collator` is implemented to intercept batches before training and force label tensors into `torch.float32` format, preventing `RuntimeError: result type Float can't be cast to the desired output type Long`

```python
def preprocess_function(examples):
    tokenized = tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=128
    )
    return tokenized
```

---

## ⚙️ Model Configuration

The model is initialized using `AutoModelForSequenceClassification` with the following specifics:

| Parameter | Value |
|-----------|-------|
| **Problem Type** | `multi_label_classification` |
| **Number of Labels** | 28 |
| **Optimization** | Mixed Precision (fp16) |
| **Loss Function** | BCEWithLogitsLoss |

```python
model = AutoModelForSequenceClassification.from_pretrained(
    "huawei-noah/TinyBERT_General_4L_312D",
    num_labels=28,
    problem_type="multi_label_classification"
)
```

---

## 🏋️ Training Details

| Configuration | Value |
|--------------|-------|
| **Hardware** | Google Colab (Tesla T4 GPU) |
| **Batch Size** | 16 (Train & Eval) |
| **Epochs** | 4 |
| **Learning Rate** | 2e-5 |
| **Metric** | F1-Micro Score |
| **Strategy** | Full fine-tuning of encoder layers |

**Training Features:**
- ✅ Mixed precision training (fp16) for faster computation
- ✅ F1-Micro Score optimized for class imbalance
- ✅ Early stopping support
- ✅ Automatic model checkpointing

---

## 🚀 Inference and Usage

Inference requires a specific post-processing step to interpret the logits:

1. **Sigmoid Activation:** Converts raw logits into probabilities (0.0 to 1.0)
2. **Thresholding:** A threshold is applied to determine active labels
   - *Default:* 0.5
   - *Recommended:* **0.25 - 0.30** (Given the model size, a lower threshold improves recall for subtle emotions)

```python
import torch

# Inference snippet
def predict_emotions(text, threshold=0.25):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    outputs = model(**inputs)
    
    # Apply sigmoid and threshold
    probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
    predictions = (probs >= threshold).astype(int)
    
    # Get predicted emotion labels
    predicted_labels = [label_names[i] for i, pred in enumerate(predictions) if pred == 1]
    return predicted_labels, probs

# Example usage
text = "I just got promoted at work! This is amazing!"
emotions, scores = predict_emotions(text)
print(f"Detected emotions: {emotions}")
```

---

## 📈 Results and Observations

### Performance Highlights

- ⚡ **Efficiency:** TinyBERT completes training significantly faster than BERT-Base
- 🎯 **Performance:** The model captures dominant sentiments (e.g., "Admiration" in achievement-related texts)
- 🔍 **Challenges:** Due to the model's small size, confidence scores for specific emotions (like "Joy") may be diluted across semantically similar labels ("Love", "Admiration")

### Key Findings

| Aspect | Observation |
|--------|-------------|
| **Training Speed** | ~4x faster than BERT-Base |
| **Model Size** | ~55% smaller than BERT-Base |
| **Inference Time** | Real-time capable on CPU |
| **Optimal Threshold** | 0.25-0.30 for best recall |

> **💡 Tip:** Adjusting the decision threshold is necessary for optimal inference results based on your use case (precision vs recall tradeoff).

---

## 📁 Repository Structure

```
finetuning-tinybert-goemotions/
│
├── notebooks/
│   └── task1_tinybert_goemotions.ipynb    # Training notebook
├── reports/
│   └── task1_encoder_classification_report.md  # Detailed report
├── README.md                               # This file
└── requirements.txt                        # Dependencies
```

---

## 🛠️ Installation & Setup

### Requirements

- Python 3.8+
- PyTorch
- Hugging Face Transformers & Datasets
- Scikit-learn & Accelerate

### Quick Start

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/finetuning-tinybert-goemotions.git
cd finetuning-tinybert-goemotions
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the training notebook:**
```bash
jupyter notebook notebooks/task1_tinybert_goemotions.ipynb
```

### Requirements File

Create a `requirements.txt` with:
```txt
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
scikit-learn>=1.2.0
accelerate>=0.20.0
numpy>=1.24.0
pandas>=2.0.0
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Hugging Face** for the Transformers library
- **Google** for the GoEmotions dataset
- **Huawei Noah's Ark Lab** for TinyBERT
- **Google Colab** for providing free GPU resources

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

## ✅ Conclusion

This project demonstrates that **TinyBERT** can effectively serve as a multi-label emotion classifier. By implementing robust data processing (custom collators) and adjusting inference thresholds, we can extract meaningful emotional insights from text even with a highly compressed model architecture suitable for resource-constrained environments.

**Key Takeaways:**
- 🎯 TinyBERT is viable for multi-label emotion classification
- ⚡ Significant speed improvements over full-sized models
- 💾 Suitable for deployment on edge devices
- 🔧 Threshold tuning is crucial for optimal performance

---

<div align="center">
  <strong>⭐ Star this repository if you find it helpful!</strong>
</div>
