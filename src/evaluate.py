import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


# Paths
X_PATH = "data/processed/X_features.npy"
Y_PATH = "data/processed/y_labels.npy"
MODEL_PATH = "models/rf_model.pkl"
SCALER_PATH = "models/feature_scaler.pkl"

GESTURE_NAMES = [
    "Open Hand",
    "Closed Hand",
    "Lateral Pinch",
    "Signalling Sign",
    "Rock Sign"
]


def main():
    # Load data
    X = np.load(X_PATH)
    y = np.load(Y_PATH)

    # Load scaler and scale features
    scaler = joblib.load(SCALER_PATH)
    X = scaler.transform(X)

    # Same split as training
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Load trained model
    model = joblib.load(MODEL_PATH)

    # Predict
    y_pred = model.predict(X_val)

    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)

    print("\nConfusion Matrix:")
    print(cm)

    # Classification Report
    print("\nPer-class Metrics:")
    print(classification_report(y_val, y_pred, target_names=GESTURE_NAMES))

    # Plot confusion matrix
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(range(5), GESTURE_NAMES, rotation=45)
    plt.yticks(range(5), GESTURE_NAMES)
    plt.colorbar()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

