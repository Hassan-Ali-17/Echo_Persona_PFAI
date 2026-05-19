import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from model import create_model

FEATURES_DIR = "data/features"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def train():
    print("Loading features...")
    try:
        X = np.load(os.path.join(FEATURES_DIR, 'X.npy'))
        y = np.load(os.path.join(FEATURES_DIR, 'y.npy'))
    except FileNotFoundError:
        print("Features not found. Please run features.py first.")
        return

    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    num_classes = len(encoder.classes_)
    y_categorical = tf.keras.utils.to_categorical(y_encoded, num_classes=num_classes)
    
    print(f"Classes found: {encoder.classes_}")
    np.save(os.path.join(MODELS_DIR, 'classes.npy'), encoder.classes_)

    # Split dataset (80/10/10) -> 80 train, 20 test (then we'll use 50% of test for validation)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    
    # Create model
    input_shape = (X_train.shape[1], X_train.shape[2])
    print(f"Input shape: {input_shape}")
    model = create_model(input_shape, num_classes=num_classes)
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(MODELS_DIR, 'best_model.h5'), 
        monitor='val_accuracy', 
        save_best_only=True, 
        mode='max', 
        verbose=1
    )
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-5)
    
    # Train
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )
    
    # Evaluate
    print("Evaluating on test set...")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print("Training complete. Model saved to models/best_model.h5")

if __name__ == "__main__":
    train()
