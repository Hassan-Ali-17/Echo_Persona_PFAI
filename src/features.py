import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_DIR = "data/raw"
FEATURES_DIR = "data/features"
os.makedirs(FEATURES_DIR, exist_ok=True)

# Fixed length (e.g., pad/truncate to 3 seconds)
MAX_LEN = 130 # approx 3 seconds at sr=22050

EMOTION_MAP = {
    '01': 'Neutral', '02': 'Calm', '03': 'Happy', '04': 'Sad',
    '05': 'Angry', '06': 'Fearful', '07': 'Disgusted', '08': 'Surprised'
}

def extract_features(file_path):
    # Load audio
    audio, sr = librosa.load(file_path, sr=22050, duration=3.0)
    
    # 1. Trim leading and trailing silence
    try:
        audio, _ = librosa.effects.trim(audio, top_db=25)
    except Exception:
        pass
        
    # 2. Apply noise reduction
    try:
        import noisereduce as nr
        audio = nr.reduce_noise(y=audio, sr=sr)
    except Exception:
        pass
        
    if len(audio) == 0:
        audio = np.zeros(100) # Fallback to prevent crash
        
    
    # 1. MFCC (40)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    
    # 2. Chroma (12)
    stft = np.abs(librosa.stft(audio))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
    
    # 3. Mel (128)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr)
    
    # 4. ZCR (1)
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    
    # 5. RMS (1)
    rms = librosa.feature.rms(y=audio)
    
    # Stack features (40+12+128+1+1 = 182 features per frame)
    features = np.vstack((mfccs, chroma, mel, zcr, rms)) # Shape: (182, T)
    
    # Pad or truncate to MAX_LEN
    if features.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - features.shape[1]
        features = np.pad(features, pad_width=((0,0), (0, pad_width)), mode='constant')
    else:
        features = features[:, :MAX_LEN]
        
    # Apply Cepstral Mean and Variance Normalization (CMVN)
    # This standardizes features per-file, canceling out microphone and volume differences
    mean = np.mean(features, axis=1, keepdims=True)
    std = np.std(features, axis=1, keepdims=True)
    features = (features - mean) / (std + 1e-8)
        
    return features.T # Transpose to (T, Features) -> (130, 182)

def process_dataset():
    features_list = []
    labels_list = []
    
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.wav')]
    print(f"Found {len(files)} audio files. Extracting features...")
    
    for file in tqdm(files):
        parts = file.split('-')
        if len(parts) >= 3:
            emotion_id = parts[2]
            if emotion_id in EMOTION_MAP:
                label = EMOTION_MAP[emotion_id]
                file_path = os.path.join(DATA_DIR, file)
                try:
                    features = extract_features(file_path)
                    features_list.append(features)
                    labels_list.append(label)
                except Exception as e:
                    print(f"Error processing {file}: {e}")
                    
    X = np.array(features_list)
    y = np.array(labels_list)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    np.save(os.path.join(FEATURES_DIR, 'X.npy'), X)
    np.save(os.path.join(FEATURES_DIR, 'y.npy'), y)
    print("Features saved to data/features/")

if __name__ == "__main__":
    process_dataset()
