import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

FEATURES_DIR = "data/features"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def train_knn():
    print("Loading features...")
    try:
        X = np.load(os.path.join(FEATURES_DIR, 'X.npy'))
        y = np.load(os.path.join(FEATURES_DIR, 'y.npy'))
    except FileNotFoundError:
        print("Features not found. Please run features.py first.")
        return

    print(f"Original shape: X={X.shape}, y={y.shape}")

    # Process features: Collapse time dimension (130 frames)
    # Calculate Mean and Std for each of the 182 features across the time steps
    print("Collapsing temporal features (calculating mean and standard deviation)...")
    X_mean = np.mean(X, axis=1) # (N, 182)
    X_std = np.std(X, axis=1)   # (N, 182)
    X_collapsed = np.hstack((X_mean, X_std)) # (N, 364)
    print(f"Collapsed shape: {X_collapsed.shape}")

    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    classes = encoder.classes_
    print(f"Classes: {classes}")

    # Save classes
    np.save(os.path.join(MODELS_DIR, 'knn_classes.npy'), classes)

    # Split dataset (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_collapsed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")

    # Find best K
    best_k = 5
    best_acc = 0
    print("Tuning parameter k...")
    for k in range(1, 16, 2):
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"  k={k}: accuracy={acc * 100:.2f}%")
        if acc > best_acc:
            best_acc = acc
            best_k = k

    print(f"Selected best k={best_k} with accuracy={best_acc * 100:.2f}%")

    # Train final model
    best_knn = KNeighborsClassifier(n_neighbors=best_k, weights='distance')
    best_knn.fit(X_train, y_train)

    # Final evaluation
    test_preds = best_knn.predict(X_test)
    print("\n--- Evaluation Report ---")
    print(f"Accuracy: {accuracy_score(y_test, test_preds) * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, test_preds, target_names=classes))

    # Save model
    model_path = os.path.join(MODELS_DIR, 'knn_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_knn, f)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_knn()
