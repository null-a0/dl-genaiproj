# Multi-Label Emotion Classification Project

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/Abhishek-A0/Emotion-Detection-RoBERTa-Abhishek)

> **Deep Learning & Generative AI Project** - Multi-label emotion classification from text using state-of-the-art transformer models and classical ML approaches.

## 🎯 Project Overview

This project tackles the challenge of multi-label emotion classification, where a single text can express multiple emotions simultaneously. We implement and compare three different approaches:

1. **RoBERTa-Large Transformer** - Fine-tuned pre-trained model
2. **BiLSTM from Scratch** - Custom deep learning architecture
3. **TF-IDF + SGD/LightGBM** - Classical machine learning baseline

### Key Features

- 🎭 Multi-label emotion detection (Joy, Sadness, Anger, Fear, Love, Surprise)
- 🚀 Three distinct modeling approaches for comparison
- 📊 Comprehensive evaluation metrics (F1, Precision, Recall)
- 🔄 Complete training and inference pipelines
- 📈 Experiment tracking with Weights & Biases
- 🌐 **Live deployment on Hugging Face Spaces**

## 🚀 Live Demo

Try out the emotion classifier in real-time:

**🔗 [Launch Demo on Hugging Face Spaces](https://huggingface.co/spaces/Abhishek-A0/Emotion-Detection-RoBERTa-Abhishek)**

Experience the RoBERTa-Large model detecting emotions from your text instantly!

## 📂 Repository Structure

```
dl-genaiproj/
├── dl-23f1001572-notebook-t32025.ipynb    # Main inference pipeline (all models)
├── roberta-large.ipynb                     # RoBERTa-Large fine-tuning
├── scratch-bilstm-model.ipynb             # Custom BiLSTM implementation
├── tfidf-sgd.ipynb                        # Classical ML approach
├── Multi-Label Emotion Classification.pdf  # Project documentation
└── README.md                              # This file
```

## 🧠 Models

### 1. RoBERTa-Large Transformer
- Pre-trained on large corpus, fine-tuned for emotion classification
- State-of-the-art performance with transfer learning
- Handles contextual understanding effectively
- **Deployed on Hugging Face Spaces**

### 2. BiLSTM from Scratch
- Custom implementation using bidirectional LSTM layers
- Word embeddings trained specifically for emotion detection
- Demonstrates deep learning fundamentals

### 3. TF-IDF + SGD/LightGBM
- Classical machine learning baseline
- Fast training and inference
- Competitive performance with traditional features

## 📊 Performance Metrics

| Model | Approach | Strengths |
|-------|----------|-----------|
| RoBERTa-Large | Transfer Learning | Best contextual understanding, deployed model |
| BiLSTM Scratch | Deep Learning | Custom architecture, good sequential processing |
| TF-IDF + SGD | Classical ML | Fast training, interpretable features |

## 🛠️ Installation

### Prerequisites
```bash
Python 3.8+
CUDA-capable GPU (recommended for transformer training)
```

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/null-a0/dl-genaiproj.git
cd dl-genaiproj

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio
pip install transformers datasets
pip install scikit-learn pandas numpy
pip install lightgbm wandb
pip install jupyter notebook
```

## 💻 Usage

### Quick Start - Inference

Run the main notebook for complete inference pipeline:

```bash
jupyter notebook dl-23f1001572-notebook-t32025.ipynb
```

### Training Individual Models

**RoBERTa-Large:**
```bash
jupyter notebook roberta-large.ipynb
```

**BiLSTM from Scratch:**
```bash
jupyter notebook scratch-bilstm-model.ipynb
```

**TF-IDF + SGD:**
```bash
jupyter notebook tfidf-sgd.ipynb
```

### Python API Example

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("Abhishek-A0/emotion-roberta-large")
tokenizer = AutoTokenizer.from_pretrained("roberta-large")

# Predict emotions
text = "I am so happy and excited about this achievement!"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.sigmoid(outputs.logits)

# Get predicted emotions
emotions = ["joy", "sadness", "anger", "fear", "love", "surprise"]
threshold = 0.5
predicted_emotions = [emotions[i] for i, score in enumerate(predictions[0]) if score > threshold]

print(f"Detected emotions: {predicted_emotions}")
```

## 📁 Dataset

The project uses a multi-label emotion dataset with the following characteristics:

- **Format**: Text with binary labels for each emotion
- **Labels**: Joy, Sadness, Anger, Fear, Love, Surprise
- **Task**: Multi-label classification (texts can have multiple emotions)
- **Split**: Training and validation sets for model evaluation

## 🔬 Methodology

### Data Preprocessing
1. Text cleaning and normalization
2. Tokenization (model-specific)
3. Label encoding for multi-label setup
4. Train/validation split with stratification

### Model Training
1. **RoBERTa-Large**: Fine-tuning with AdamW optimizer, learning rate scheduling
2. **BiLSTM**: Custom architecture with embedding layer, bidirectional LSTM, dropout
3. **TF-IDF**: Feature extraction followed by classifier training

### Evaluation
- Multi-label metrics: Micro/Macro F1, Hamming Loss
- Per-class performance analysis
- Confusion matrix visualization
- Error analysis

## 📈 Experiment Tracking

All experiments are tracked using Weights & Biases:

- Training/validation loss curves
- Metric evolution over epochs
- Hyperparameter configurations
- Model checkpoints

## 🌐 Deployment

The RoBERTa-Large model is deployed on Hugging Face Spaces for easy access and testing:

**Live Demo**: [https://huggingface.co/spaces/Abhishek-A0/Emotion-Detection-RoBERTa-Abhishek](https://huggingface.co/spaces/Abhishek-A0/Emotion-Detection-RoBERTa-Abhishek)

### Deployment Features
- Real-time emotion prediction
- Interactive web interface
- Support for multiple text inputs
- Instant results with confidence scores

```

## 👤 Author

**Abhishek Saha**
- Roll Number: 23f1001572
- GitHub: [@null-a0](https://github.com/null-a0)
- Hugging Face: [@Abhishek-A0](https://huggingface.co/Abhishek-A0)
- Project: DL & GenAI - September 2025

## 📄 License

This project is part of an academic assignment. Please contact the author for usage permissions.

## 🙏 Acknowledgments

- Hugging Face for transformer models and deployment platform
- PyTorch team for the deep learning framework
- Weights & Biases for experiment tracking
- Course instructors and peers for guidance and feedback

## 🔗 Additional Resources

- [Project Documentation PDF](Multi-Label%20Emotion%20Classification.pdf)
- [Live Demo - Hugging Face Space](https://huggingface.co/spaces/Abhishek-A0/Emotion-Detection-RoBERTa-Abhishek)
- [GitHub Repository](https://github.com/null-a0/dl-genaiproj)

---

**⭐ If you find this project useful, please consider giving it a star!**
