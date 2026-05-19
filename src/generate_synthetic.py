import os
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

DATA_DIR = "data/raw"
os.makedirs(DATA_DIR, exist_ok=True)

# 8 Emotions
EMOTIONS = ['01', '02', '03', '04', '05', '06', '07', '08']
NUM_SAMPLES = 20
SR = 22050
DURATION = 3.0
T = int(SR * DURATION)

def generate_tone(emotion_id):
    t = np.linspace(0, DURATION, T, endpoint=False)
    
    if emotion_id == '01': # Neutral
        freq = 300
        noise_level = 0.05
    elif emotion_id == '02': # Shy
        freq = 250
        noise_level = 0.02
    elif emotion_id == '03': # Happy
        freq = 500
        noise_level = 0.1
    elif emotion_id == '04': # Sad
        freq = 200
        noise_level = 0.01
    elif emotion_id == '05': # Angry
        freq = 600
        noise_level = 0.3
    elif emotion_id == '06': # Fearful
        freq = 550
        noise_level = 0.2
    elif emotion_id == '07': # Disgusted
        freq = 150
        noise_level = 0.15
    elif emotion_id == '08': # Surprised
        freq = 700
        noise_level = 0.05
    else:
        freq = 400
        noise_level = 0.1
        
    # Generate base tone
    audio = np.sin(2 * np.pi * freq * t)
    # Add harmonics
    audio += 0.5 * np.sin(2 * np.pi * freq * 2 * t)
    # Add noise
    noise = np.random.normal(0, noise_level, T)
    audio = audio + noise
    # Normalize
    audio = audio / np.max(np.abs(audio))
    return audio

def create_synthetic_dataset():
    print("Generating synthetic audio dataset...")
    for em in tqdm(EMOTIONS):
        for i in range(1, NUM_SAMPLES + 1):
            audio = generate_tone(em)
            # Filename format similar to RAVDESS: 03-01-[EMOTION]-01-01-01-[ACTOR].wav
            filename = f"03-01-{em}-01-01-01-{i:02d}.wav"
            filepath = os.path.join(DATA_DIR, filename)
            # Convert to 16-bit PCM
            audio_16bit = np.int16(audio * 32767)
            wavfile.write(filepath, SR, audio_16bit)
            
    print(f"Generated {len(EMOTIONS) * NUM_SAMPLES} synthetic audio files in {DATA_DIR}.")

if __name__ == "__main__":
    create_synthetic_dataset()
