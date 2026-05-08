<div align="center">

```
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║               ███████╗ ██████╗██╗  ██╗ ██████╗                    ║
║               ██╔════╝██╔════╝██║  ██║██╔═══██╗                   ║
║               █████╗  ██║     ███████║██║   ██║                   ║
║               ██╔══╝  ██║     ██╔══██║██║   ██║                   ║
║               ███████╗╚██████╗██║  ██║╚██████╔╝                   ║
║               ╚══════╝ ╚═════╝╚═╝  ╚═╝ ╚═════╝                    ║
║                                                                   ║
║   ██████╗ ███████╗██████╗ ███████╗ ██████╗ ███╗  ██╗ █████╗       ║
║   ██╔══██╗██╔════╝██╔══██╗██╔════╝██╔═══██╗████╗ ██║██╔══██╗      ║
║   ██████╔╝█████╗  ██████╔╝███████╗██║   ██║██╔██╗██║███████║      ║
║   ██╔═══╝ ██╔══╝  ██╔══██╗╚════██║██║   ██║██║╚████║██╔══██║      ║
║   ██║     ███████╗██║  ██║███████║╚██████╔╝██║ ╚███║██║  ██║      ║
║   ╚═╝     ╚══════╝╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚══╝╚═╝  ╚═╝      ║
║                                                                   ║
║      🎙️  Personality & Emotion Detection from Audio Signals      ║
╚═══════════════════════════════════════════════════════════════════╝
```

</div>

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Librosa](https://img.shields.io/badge/Librosa-Audio%20Processing-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

</div>

---

## 👥 Team

## 👥 Team
 
| Name | ID | GitHub |
|---|---|---|
| Hassan Ali Shah | 24040 | [@Hassan-Ali-17](https://github.com/Hassan-Ali-17) |
| Ahsen Ali | 24056 | [@ahsen24056ali](https://github.com/ahsen24056ali) |
| Abdul Moeed | 24140 | [@abdul888-888](https://github.com/abdul888-888) |
| Hamza Abbas | 24036 | [@Hammaabbas1234](https://github.com/Hammaabbas1234) |
 
**Course:** Principles of AI (PFAI) · Deep Learning Project · 4th Semester

**Course:** Principles of AI (PFAI) · Deep Learning Project · 4th Semester

---

## 📌 Table of Contents

- [Overview](#overview)
- [Emotion Classes](#emotion-classes)
- [Dataset — PEAD](#dataset--pead)
- [System Pipeline](#system-pipeline)
- [Model Architecture](#model-architecture)
- [Feature Extraction](#feature-extraction)
- [Results & Evaluation](#results--evaluation)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [References](#references)

---

## Overview

Human emotion detection from audio is one of AI's most challenging problems — emotions are subjective, audio quality varies wildly, and emotional states frequently overlap. **EchoPersona** tackles those challenges with a full deep learning pipeline:

- 🎤 **Self-collected audio dataset** (PEAD) — 150 real recordings, 30 speakers
- 🔊 **Rich acoustic features** — MFCC, Mel Spectrogram, Chroma, ZCR, RMS
- 🧠 **Hybrid deep learning model** — CNN + Bidirectional LSTM + Attention
- 🚦 **Confidence-aware post-processing** — thresholding, sliding window, temporal smoothing

**Applications:** Mental health monitoring · Human-computer interaction · Call center analytics · EdTech · Automotive safety · Social robotics

---

## Emotion Classes

| Label | Description | Samples |
|---|---|---|
| 😊 Happy | Joyful, excited, enthusiastic speech | 20 |
| 😢 Sad | Sorrowful, melancholic, low-energy speech | 20 |
| 😡 Angry | Aggressive, loud, fast-paced speech | 20 |
| 😰 Shy / Nervous | Hesitant, quiet, stuttering speech | 20 |
| 😨 Fearful | Trembling voice, rapid speech, anxious tone | 15 |
| 🤢 Disgusted | Contemptuous, repulsed vocal expressions | 15 |
| 😲 Surprised | Sudden exclamation, pitch spikes | 15 |
| 😐 Neutral | Flat, emotionless, baseline speech | 25 |
| | **TOTAL** | **150** |

---

## Dataset — PEAD

**PEAD (Personality & Emotion Audio Dataset)** is a custom self-collected dataset built specifically for EchoPersona.

```
╔════════════════════════╦══════════════════════════════════════════╗
║ Attribute              ║ Value                                    ║
╠════════════════════════╬══════════════════════════════════════════╣
║ Dataset Name           ║ PEAD — Personality & Emotion Audio       ║
║ Format                 ║ WAV (uncompressed, lossless)             ║
║ Sample Rate            ║ 44,100 Hz (CD Quality)                   ║
║ Bit Depth              ║ 16-bit                                   ║
║ Language               ║ English (+ some Urdu/bilingual)          ║
║ Unique Speakers        ║ 30 (18–45 yrs, 50% male / 50% female)    ║
║ Avg Duration           ║ 4–8 seconds per sample                   ║
║ Total Duration         ║ ~15–20 minutes                           ║
║ Total Size             ║ ~500 MB – 1.2 GB (uncompressed)          ║
╚════════════════════════╩══════════════════════════════════════════╝
```

### Collection Methodology

1. **Recruit** — 30 volunteers aged 18–45, signed informed consent, anonymized by Speaker ID
2. **Script** — Standardized emotionally loaded sentences per class (5 sentences/speaker/class)
3. **Record** — 50% professional (USB condenser mic, Audacity) + 50% naturalistic (smartphone in bedroom/office/outdoors)
4. **QC** — Accepted only clips with SNR > 15 dB, genuine emotion, no clipping; ~10–15% re-recorded

---

## System Pipeline

```
╔══════════════════════════════════════════════════════════════╗
║                   ECHOPERSONA PIPELINE                       ║
╚══════════════════════════════════════════════════════════════╝

      Raw Audio (WAV)
            │
            ▼
  ╔═════════════════════╗
  ║   1. PREPROCESSING  ║  Resample → 22,050 Hz
  ║                     ║  Trim silence · Noise reduction
  ║                     ║  Peak normalize · Pad/truncate 5s
  ╚══════════╦══════════╝
             │
             ▼
  ╔═════════════════════╗
  ║   2. AUGMENTATION   ║  5× factor
  ║                     ║  Gaussian noise
  ║                     ║  Time stretch ×0.9 / ×1.1
  ║                     ║  Pitch shift ±2 semitones
  ║                     ║  150 → 900 samples
  ╚══════════╦══════════╝
             │
             ▼
  ╔═════════════════════╗
  ║   3. FEATURE        ║  MFCC (40 + Δ + ΔΔ) → 120 × T
  ║      EXTRACTION     ║  Mel Spectrogram      → 128 × T
  ║                     ║  Chroma               →  12 × T
  ║                     ║  ZCR + RMS            →   2 × T
  ║                     ║  ─────────────────────────────
  ║                     ║  Combined Feature     → 134 × T
  ╚══════════╦══════════╝
             │
             ▼
  ╔═════════════════════╗
  ║   4. MODEL          ║  CNN + BiLSTM + Attention
  ║      TRAINING       ║  Adam lr=0.001 · 100 epochs
  ║                     ║  80/10/10 speaker-independent
  ╚══════════╦══════════╝
             │
             ▼
  ╔═════════════════════╗
  ║   5. POST-          ║  Confidence threshold 60%
  ║      PROCESSING     ║  Sliding window majority vote
  ║                     ║  Temporal smoothing (window=5)
  ╚══════════╦══════════╝
             │
             ▼
      Emotion Label + Confidence Score
```

---

## Model Architecture

```
╔══════════════════════════════════════════════════════════════╗
║              ECHOPERSONA MODEL ARCHITECTURE                  ║
╚══════════════════════════════════════════════════════════════╝

   Input: (134 features × T time frames)
                 │
                 ▼
   ╔═════════════════════════════════╗
   ║           CNN BLOCK             ║
   ║  Conv1D(64,  k=3) → BN → ReLU   ║
   ║  Conv1D(128, k=3) → BN → ReLU   ║
   ║  MaxPool(2) → Dropout(0.3)      ║
   ╚════════════════╦════════════════╝
                    │
                    ▼
   ╔═════════════════════════════════╗
   ║          BiLSTM BLOCK           ║
   ║  BiLSTM(256, return_seq=True)   ║
   ║  BiLSTM(128, return_seq=True)   ║
   ║  Dropout(0.4)                   ║
   ╚════════════════╦════════════════╝
                    │
                    ▼
   ╔═════════════════════════════════╗
   ║        ATTENTION LAYER          ║
   ║  Dense(1, tanh) → Softmax       ║
   ║  Weighted Sum of time frames    ║
   ╚════════════════╦════════════════╝
                    │
                    ▼
   ╔═════════════════════════════════╗
   ║      FULLY CONNECTED LAYERS     ║
   ║  Dense(256) → BN → ReLU         ║
   ║  Dropout(0.5)                   ║
   ║  Dense(128) → Dropout(0.3)      ║
   ╚════════════════╦════════════════╝
                    │
                    ▼
   ╔═════════════════════════════════╗
   ║          OUTPUT LAYER           ║
   ║     Dense(8) → Softmax          ║
   ║   Probability over 8 classes    ║
   ╚═════════════════════════════════╝
```

**Training config:** Adam (lr=0.001) · Categorical cross-entropy · Batch size 32 · Early stopping (patience=15) · ReduceLROnPlateau (patience=7) · ModelCheckpoint

> ⚠️ Split is **speaker-independent** — the same speaker never appears in both train and test sets.

---

## Feature Extraction

| Feature | Shape | What it captures |
|---|---|---|
| MFCC (40 coeffs + Δ + ΔΔ) | 120 × T | Vocal tract shape; frequency energy distribution |
| Mel Spectrogram (128 banks) | 128 × T | Time-frequency content; CNN-friendly 2D representation |
| Chroma (12 pitch classes) | 12 × T | Pitch variation; harmonic patterns |
| Zero Crossing Rate (ZCR) | 1 × T | Noisiness; fricative vs. voiced speech |
| Root Mean Square Energy | 1 × T | Overall loudness and energy envelope |
| **Combined** | **134 × T** | |

---

## Results & Evaluation

Metrics used: **Accuracy · Precision · Recall · F1-Score · Confusion Matrix · ROC-AUC**

```
╔══════════════════════════════════════════════╗
║             EVALUATION RESULTS               ║
╠══════════════════════╦═══════════════════════╣
║ Metric               ║ Score                 ║
╠══════════════════════╬═══════════════════════╣
║ Accuracy             ║ TBD after training    ║
║ Weighted F1          ║ TBD after training    ║
║ ROC-AUC (avg)        ║ TBD after training    ║
╚══════════════════════╩═══════════════════════╝
```

> ⚙️ Will be updated once training is complete.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/EchoPersona.git
cd EchoPersona

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Key dependencies:**

```
tensorflow>=2.10
librosa>=0.9
noisereduce
numpy
pandas
scikit-learn
matplotlib
seaborn
jupyter
```

---

## Usage

```bash
# 1. Preprocess & augment audio
python src/preprocess.py --input data/raw/ --output data/processed/

# 2. Extract features
python src/features.py --input data/processed/ --output data/features/

# 3. Train the model
python src/train.py --features data/features/ --epochs 100 --batch-size 32

# 4. Evaluate
python src/evaluate.py --model models/best_model.h5 --test data/features/test/

# 5. Predict on new audio
python src/predict.py --model models/best_model.h5 --audio path/to/audio.wav
```

---

## Project Structure

```
╔══════════════════════════════════════════════════════╗
║                   EchoPersona/                       ║
╠══════════════════════════════════════════════════════╣
║  📁 data/                                            ║
║  ├── 📁 raw/              Original WAV recordings    ║
║  ├── 📁 processed/        Cleaned & normalized       ║
║  └── 📁 features/         Extracted feature arrays   ║
║                                                      ║
║  📁 models/                                          ║
║  └── 🧠 best_model.h5     Saved model weights        ║
║                                                      ║
║  📁 notebooks/                                       ║
║  ├── 01_preprocessing.ipynb                          ║
║  ├── 02_feature_extraction.ipynb                     ║
║  ├── 03_model_training.ipynb                         ║
║  └── 04_evaluation.ipynb                             ║
║                                                      ║
║  📁 src/                                             ║
║  ├── preprocess.py                                   ║
║  ├── features.py                                     ║
║  ├── model.py                                        ║
║  ├── train.py                                        ║
║  ├── evaluate.py                                     ║
║  └── predict.py                                      ║
║                                                      ║
║  📄 requirements.txt                                 ║
║  📄 README.md                                        ║
╚══════════════════════════════════════════════════════╝
```

---

## References

| Paper | Authors | Year | Key Contribution |
|---|---|---|---|
| Speech Emotion Recognition Using Deep Learning | Zhao, Mao & Chen | 2021 | CNN + LSTM dual-branch architecture |
| Automatic Personality Perception from Audio-Visual Data | Palmero et al. | 2022 | OCEAN personality from multimodal data |
| Attention-Based BiLSTM for SER | Li, Zhao & Kawahara | 2022 | Attention mechanism for salient frame selection |
| Wav2Vec 2.0 for Emotion Recognition | Pepino, Riera & Ferrer | 2023 | Self-supervised pre-training on raw waveforms |

---

<div align="center">

```
╔══════════════════════════════════════════╗
║    Made with ❤️  by EchoPersona Team     ║
║        PFAI Course · 4th Semester        ║
╚══════════════════════════════════════════╝
```

</div>
