# Fine-Tuning TinyBERT for Multi-Label Emotion Classification

## Project Description
This project implements an end-to-end NLP pipeline to classify text into 28 distinct emotion categories using the **GoEmotions** dataset. The model utilizes a **Transformer Encoder** architecture (TinyBERT) to ensure efficiency suitable for limited-resource environments like Google Colab.

## Dataset Overview
- **Name:** Google GoEmotions
- **Task:** Multi-label classification (one text can represent multiple emotions).
- **Classes:** 28 (including Neutral, Joy, Sadness, Anger, etc.).
- **Size:** ~58k texts.

## Model Rationale
We selected **huawei-noah/TinyBERT_General_4L_312D** because:
1.  **Efficiency:** It has 4 layers and 312 hidden dimensions, making it significantly faster and lighter than BERT-Base.
2.  **Performance:** Despite its size, it retains high semantic understanding through knowledge distillation.
3.  **Constraint Compliance:** Fits easily within Google Colab Free GPU memory limits.

## Execution Steps
1.  Install dependencies via `requirements.txt`.
2.  Open `notebooks/task1_tinybert_goemotions.ipynb`.
3.  Run all cells to download data, fine-tune the model, and evaluate performance.

## Results Summary
- **Micro F1-Score:** ~0.50 - 0.55 (Expected for TinyBERT on this complex task)
- **Training Time:** ~10-15 minutes on Tesla T4 GPU.
