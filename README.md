# Heart Murmur Detection Using Audio Signals & Deep Learning

An end-to-end machine learning project that classifies heart sounds to detect murmurs using audio preprocessing, MFCC feature extraction, and an LSTM-based deep learning model — deployed as an interactive Streamlit web application.

---

## Project Overview

Heart murmurs are abnormal sounds produced during a heartbeat cycle and can be early indicators of serious cardiovascular conditions. Timely detection plays a critical role in diagnosis and treatment planning.

This project applies audio signal processing and deep learning to automate the classification of heart sounds, covering the complete ML pipeline:

- Data loading and preprocessing
- Feature extraction (MFCC)
- Model training with class imbalance handling
- Model deployment via Streamlit
- Cloud-based model hosting on Hugging Face Hub

---

## Objective

To design and deploy a deep learning system capable of analyzing raw heart sound recordings and accurately classifying them to assist in automated murmur detection.
---

## Dataset

| Property | Details |
|----------|---------|
| **Name** | Heartbeat Sound Dataset |
| **Source** | Kaggle (Open Source) |
| **Link** | [View Dataset](https://www.kaggle.com/datasets/abdallahaboelkhair/heartbeat-sound) |
| **Format** | `.wav` audio files |

**Classes Used:**
- Normal
- Murmur
- Artifact
- Extrahls
- Extrastole

> Unlabelled audio files are used exclusively for testing and inference.

---

## Tools & Technologies

| Category | Tools |
|----------|-------|
| Language | Python |
| Audio Processing | Librosa, NumPy |
| Deep Learning | TensorFlow, Keras |
| Model Architecture | LSTM |
| Feature Extraction | MFCC |
| Web Framework | Streamlit |
| Model Hosting | Hugging Face Hub |
| Deployment | Streamlit Cloud |
| Version Control | Git & GitHub |

---

## System Architecture

```
Audio Input (.wav / .mp3)
        ↓
Audio Preprocessing
(Sampling Rate Normalization + Duration Padding)
        ↓
MFCC Feature Extraction
        ↓
LSTM Deep Learning Model
        ↓
Prediction Output
        ↓
Streamlit Web Interface

```

## Project Workflow

1. Load heart sound audio files from the dataset
2. Normalize audio to a standard sampling rate (22,050 Hz)
3. Pad or trim each file to a fixed duration (10 seconds)
4. Extract MFCC features from the processed audio
5. Compute class weights to address label imbalance
6. Train the LSTM model using weighted loss
7. Evaluate model performance on a held-out test set
8. Upload the trained model to Hugging Face Hub
9. Deploy the inference pipeline via Streamlit

## Handling Class Imbalance

The dataset has an uneven distribution of class samples. Rather than resampling the data, **class weights** are computed and applied during training, causing the model to penalize misclassifications on minority classes more heavily.

**Why this approach?**
- Retains all original data without synthetic augmentation
- Reduces the risk of overfitting
- Particularly well-suited for medical and clinical datasets


## 🌐 Web Application (Streamlit)

The Streamlit app allows users to:
- Upload heart sound audio files (`.wav` / `.mp3`)  
- Visualize the audio waveform  
- Run real-time murmur prediction  
- View model confidence scores  

The trained model is **loaded dynamically from Hugging Face Hub** for scalability and version control.


## Project Structure

```
├── app.py                  # Streamlit application entry point
├── config.py               # Project-wide configuration constants
├── requirements.txt        # Python dependencies
├── model/
│   └── loader.py           # Model loading from Hugging Face Hub
├── audio/
│   └── preprocessing.py    # Audio normalization and feature extraction
├── ui/
│   └── visualizations.py   # Waveform and prediction display components
├── utils/
│   └── logger.py           # Logging utilities
└── README.md
```




## 📌 Key Learning Outcomes

- Working with audio data for ML  
- Audio preprocessing techniques  
- MFCC feature extraction  
- Sequence modeling using LSTM  
- Handling class imbalance  
- Model deployment with Streamlit  
- Cloud-based model hosting with Hugging Face  



## 🔮 Future Improvements

- Add CNN-based spectrogram models  
- Improve evaluation using precision & recall  
- Add model explainability  
- Support real-time audio recording  
- Provide REST API for predictions  



## Acknowledgements

- [Kaggle](https://www.kaggle.com) — for providing the open-source heartbeat dataset
- [Librosa](https://librosa.org) & [TensorFlow](https://www.tensorflow.org) — for their excellent open-source libraries
- [Streamlit](https://streamlit.io) & [Hugging Face](https://huggingface.co) — for making deployment accessible


> If you found this project useful, feel free to ⭐ the repository!
