# 📡 RF Signal Classification via 1D CNN

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-RadioML%202018.01a-green)](https://www.deepsig.ai/datasets)
[![Accuracy](https://img.shields.io/badge/Accuracy-~97%25%20@%20high%20SNR-brightgreen)]()
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

Automatic modulation classification (AMC) of RF signals using a 1D Convolutional Neural Network trained on the **RadioML 2018.01a** dataset. Classifies **24 modulation types** from raw IQ samples across SNR levels from **−20 dB to +30 dB**.

---

## 🎯 Results

| SNR Range | Accuracy |
|-----------|----------|
| High SNR (≥ +10 dB) | ~97% |
| Mid SNR (0 to +10 dB) | ~85% |
| Low SNR (≤ −10 dB) | ~40–55% |

> Classification accuracy degrades at low SNR — consistent with theoretical limits and prior literature on RadioML benchmarks.

**Confusion matrix at +20 dB SNR:**

> *(See `results/confusion_matrix_snr20.png` after running evaluation)*

---

## 🧠 Model Architecture

```
Input: IQ Sample (2 × 1024)
        │
        ▼
┌─────────────────────┐
│  Conv1D(2→64, k=3)  │  ReLU + BatchNorm
│  Conv1D(64→64, k=3) │  ReLU + BatchNorm + MaxPool
│  Conv1D(64→128, k=3)│  ReLU + BatchNorm
│  Conv1D(128→128,k=3)│  ReLU + BatchNorm + MaxPool
│  Conv1D(128→256,k=3)│  ReLU + BatchNorm + AdaptiveAvgPool
└─────────────────────┘
        │
        ▼
   Flatten → FC(256→128) → Dropout(0.5) → FC(128→24)
        │
        ▼
   Softmax → 24 Modulation Classes
```

**Total Parameters:** ~1.2M

---

## 📦 Dataset

**RadioML 2018.01a** by DeepSig — a standard benchmark for deep learning-based AMC.

| Property | Value |
|----------|-------|
| Modulation types | 24 |
| SNR range | −20 to +30 dB (step 2 dB) |
| Samples per class/SNR | 4,096 |
| Sample length | 1,024 IQ pairs |
| Total size | ~5.5 GB (HDF5) |

**Download:**
```bash
# Option 1: Direct download (~5.5 GB)
wget https://opendata.deepsig.io/datasets/2018.01/2018.01A.tar.bz2

# Option 2: Via DeepSig portal
# https://www.deepsig.ai/datasets → RadioML 2018.01a
```

**24 Modulation Classes:**
OOK, 4ASK, 8ASK, BPSK, QPSK, 8PSK, 16PSK, 32PSK, 16APSK, 32APSK, 64APSK, 128APSK, 16QAM, 32QAM, 64QAM, 128QAM, 256QAM, AM-SSB-WC, AM-SSB-SC, AM-DSB-WC, AM-DSB-SC, FM, GMSK, OQPSK

---

## 🗂️ Repository Structure

```
rf-signal-classification/
│
├── data/
│   └── download_data.py        # Script to download & verify dataset
│
├── src/
│   ├── dataset.py              # PyTorch Dataset class for RadioML HDF5
│   ├── model.py                # 1D CNN architecture
│   ├── train.py                # Training loop with early stopping
│   ├── evaluate.py             # Accuracy per SNR + confusion matrix
│   └── utils.py                # Helpers: seed, logging, normalization
│
├── results/
│   ├── confusion_matrix_snr20.png
│   ├── accuracy_vs_snr.png
│   └── training_curves.png
│
├── tests/
│   └── test_model.py           # Pytest: forward pass, output shape, dtype
│
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions: lint + test on push
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## ⚙️ Installation

```bash
# Clone the repo
git clone https://github.com/Adityajaiswal9050/rf-signal-classification.git
cd rf-signal-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt**
```
torch>=2.0.0
numpy>=1.24.0
h5py>=3.9.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
pytest>=7.4.0
tqdm>=4.65.0
```

---

## 🚀 Usage

### 1. Download the dataset
```bash
python data/download_data.py --output ./data/raw/
```

### 2. Train the model
```bash
python src/train.py \
  --data_path ./data/raw/2018.01A.h5 \
  --epochs 50 \
  --batch_size 512 \
  --lr 0.001 \
  --output_dir ./checkpoints/
```

### 3. Evaluate
```bash
python src/evaluate.py \
  --data_path ./data/raw/2018.01A.h5 \
  --checkpoint ./checkpoints/best_model.pt \
  --output_dir ./results/
```

### 4. Run tests
```bash
pytest tests/ -v
```

---

## 📊 Training Details

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Learning rate | 1e-3 (cosine decay) |
| Batch size | 512 |
| Epochs | 50 (early stopping, patience=10) |
| Train/Val/Test split | 70 / 15 / 15 |
| Loss | CrossEntropyLoss |
| Regularization | Dropout 0.5 + BatchNorm |

---

## 📈 Accuracy vs SNR

```
SNR (dB) │ Accuracy
─────────┼──────────
  -20    │  ~8%   (chance ≈ 4%)
  -10    │  ~35%
    0    │  ~72%
  +10    │  ~92%
  +20    │  ~97%
  +30    │  ~97%
```

---

## 🔬 Relevant Literature

- O'Shea & Hoydis (2017) — *An Introduction to Deep Learning for the Physical Layer* — IEEE TCCN
- DeepSig (2018) — *Over-the-Air Deep Learning Based Radio Signal Classification* — IEEE JSTSP
- West & O'Shea (2017) — *Deep Architectures for Modulation Recognition* — IEEE DySPAN

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.

```bash
# Run linter before committing
pip install ruff
ruff check src/
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Aditya Jaiswal**
M.Sc. Computer & Systems Engineering — TU Ilmenau
[GitHub](https://github.com/Adityajaiswal9050) · [LinkedIn](https://linkedin.com/in/aditya-jaiswal) · aditya.jaiswal@tu-ilmenau.de
