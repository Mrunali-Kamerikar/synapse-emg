import os
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler


# Paths
FEATURES_PATH = "data/processed/X_features.npy"
LABELS_PATH = "data/processed/y_labels.npy"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "feature_scaler.pkl")


def main():
    print("Loading dataset...")
    X = np.load(FEATURES_PATH)
    y = np.load(LABELS_PATH)

    print("Dataset shape:", X.shape)

    # -----------------------------
    # Feature scaling (IMPORTANT)
    # -----------------------------
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Random Forest (best-performing version)
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    print("Training model...")
    model.fit(X_train, y_train)

    # Validation
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="macro")

    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Validation F1 Score: {f1:.4f}")

    # Save model and scaler
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print("Model saved to", MODEL_PATH)
    print("Scaler saved to", SCALER_PATH)


if __name__ == "__main__":
    main()

