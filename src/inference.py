import joblib
import numpy as np

from preprocess import preprocess_csv

MODEL_PATH = "models/rf_model.pkl"
SCALER_PATH = "models/feature_scaler.pkl"

GESTURE_MAP = {
    0: "Open Hand",
    1: "Closed Hand",
    2: "Lateral Pinch",
    3: "Signalling Sign",
    4: "Rock Sign"
}


def predict_gesture(csv_path):
    print("Reading file:", csv_path)

    # Load model and scaler
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Preprocess CSV
    features = preprocess_csv(csv_path)
    print("Raw feature vector (first 5 values):", features[:5])

    # Scale features
    features = scaler.transform(features.reshape(1, -1))

    # Predict probabilities
    probs = model.predict_proba(features)[0]
    print("Class probabilities:", probs)

    prediction = np.argmax(probs)
    return GESTURE_MAP[prediction]


if __name__ == "__main__":
    test_csv = "data/Synapse_Dataset/Session2/session2_subject_12/gesture03_trial04.csv"
    gesture = predict_gesture(test_csv)
    print("Predicted Gesture:", gesture)

