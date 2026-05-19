import os
import sys
import numpy as np
import librosa
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf

# Add src to path to import features logic
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from features import extract_features, MAX_LEN

app = FastAPI(title="EchoPersona API")

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and classes
MODEL_PATH = os.path.join("models", "best_model.h5")
CLASSES_PATH = os.path.join("models", "classes.npy")

model = None
classes = None

@app.on_event("startup")
def load_resources():
    global model, classes
    if os.path.exists(MODEL_PATH) and os.path.exists(CLASSES_PATH):
        model = tf.keras.models.load_model(MODEL_PATH, safe_mode=False)
        classes = np.load(CLASSES_PATH)
        print("Model and classes loaded successfully.")
    else:
        print("Warning: Model or classes not found. Training needs to be run first.")

@app.post("/api/predict")
async def predict_emotion(file: UploadFile = File(...)):
    if not model or classes is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Please train the model first.")
        
    if not file.filename.endswith(('.wav', '.mp3', '.flac', '.webm', '.ogg')):
        raise HTTPException(status_code=400, detail="Unsupported file format.")
        
    try:
        # Save temp file
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())
            
        # Extract features
        features = extract_features(temp_path) # Shape: (T, Features)
        
        # Add batch dimension
        features = np.expand_dims(features, axis=0) # Shape: (1, T, Features)
        
        # Predict
        predictions = model.predict(features)[0]
        predicted_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_idx])
        emotion = classes[predicted_idx]
        
        # Optional: post-processing threshold
        if confidence < 0.30:
            emotion = "Uncertain"
            
        # Clean up
        os.remove(temp_path)
        
        return JSONResponse({
            "emotion": emotion,
            "confidence": confidence,
            "all_scores": {classes[i]: float(predictions[i]) for i in range(len(classes))}
        })
        
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

# Mount frontend
frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))
os.makedirs(frontend_path, exist_ok=True)
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
