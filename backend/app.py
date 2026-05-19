import os
import sys
import numpy as np
import librosa
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Add src to path to import features logic
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
# pyright: ignore [reportMissingImports]
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

# Load models and classes
import pickle

MODEL_PATH = os.path.join("models", "best_model.h5")
CLASSES_PATH = os.path.join("models", "classes.npy")
KNN_MODEL_PATH = os.path.join("models", "knn_model.pkl")
KNN_CLASSES_PATH = os.path.join("models", "knn_classes.npy")

model = None
classes = None
knn_model = None
knn_classes = None

@app.on_event("startup")
def load_resources():
    global model, classes, knn_model, knn_classes
    
    # Try loading Deep Learning model (may fail if tensorflow has issues on python 3.13)
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(CLASSES_PATH):
            import tensorflow as tf
            model = tf.keras.models.load_model(MODEL_PATH, safe_mode=False)
            classes = np.load(CLASSES_PATH)
            print("Deep Learning model and classes loaded successfully.")
        else:
            print("Warning: Deep Learning model or classes not found. Training needs to be run first.")
    except Exception as e:
        print(f"Warning: Could not load Deep Learning model. Error: {e}")
        print("Deep Learning features will be disabled. You can still use the KNN model.")
        
    if os.path.exists(KNN_MODEL_PATH) and os.path.exists(KNN_CLASSES_PATH):
        with open(KNN_MODEL_PATH, 'rb') as f:
            knn_model = pickle.load(f)
        knn_classes = np.load(KNN_CLASSES_PATH)
        print("KNN model and classes loaded successfully.")
    else:
        print("Warning: KNN model or classes not found.")

@app.post("/api/predict")
async def predict_emotion(file: UploadFile = File(...), model_type: str = "dl"):
    if model_type == "knn":
        if not knn_model or knn_classes is None:
            raise HTTPException(status_code=503, detail="KNN Model is not loaded. Please train it first.")
    else:
        if not model or classes is None:
            raise HTTPException(status_code=503, detail="Deep Learning model is not loaded. Please train it first.")
        
    if not file.filename.endswith(('.wav', '.mp3', '.flac', '.webm', '.ogg')):
        raise HTTPException(status_code=400, detail="Unsupported file format.")
        
    temp_path = f"temp_{file.filename}"
    try:
        # Save temp file
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())
            
        # Extract features (Shape: (T, Features) -> (130, 182))
        features = extract_features(temp_path)
        
        if model_type == "knn":
            # Collapse features using mean and standard deviation over time frames
            f_mean = np.mean(features, axis=0) # (182,)
            f_std = np.std(features, axis=0)   # (182,)
            f_collapsed = np.hstack((f_mean, f_std)) # (364,)
            f_collapsed = np.expand_dims(f_collapsed, axis=0) # (1, 364)
            
            # Predict
            predictions = knn_model.predict_proba(f_collapsed)[0]
            predicted_idx = np.argmax(predictions)
            confidence = float(predictions[predicted_idx])
            emotion = knn_classes[predicted_idx]
            current_classes = knn_classes
        else:
            # Add batch dimension
            features = np.expand_dims(features, axis=0) # Shape: (1, T, Features)
            
            # Predict
            predictions = model.predict(features)[0]
            predicted_idx = np.argmax(predictions)
            confidence = float(predictions[predicted_idx])
            emotion = classes[predicted_idx]
            current_classes = classes
            
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return JSONResponse({
            "emotion": emotion,
            "confidence": confidence,
            "all_scores": {current_classes[i]: float(predictions[i]) for i in range(len(current_classes))},
            "model_used": "KNN" if model_type == "knn" else "Deep Learning"
        })
        
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

# Mount frontend
frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))
os.makedirs(frontend_path, exist_ok=True)
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
