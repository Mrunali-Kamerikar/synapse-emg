import os
import numpy as np
from preprocess import preprocess_csv
from dataset import get_label_from_filename

DATASET_ROOT = "data/Synapse_Dataset"
OUTPUT_DIR = "data/processed"

def build_feature_dataset():
    X = []
    y = []

    print("Starting full preprocessing of dataset...")

    for session in os.listdir(DATASET_ROOT):
        session_path = os.path.join(DATASET_ROOT, session)
        if not os.path.isdir(session_path):
            continue

        for subject in os.listdir(session_path):
            subject_path = os.path.join(session_path, subject)
            if not os.path.isdir(subject_path):
                continue

            for file in os.listdir(subject_path):
                if not file.endswith(".csv"):
                    continue

                file_path = os.path.join(subject_path, file)

                features = preprocess_csv(file_path)
                label = get_label_from_filename(file)

                X.append(features)
                y.append(label)

    X = np.array(X)
    y = np.array(y)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, "X_features.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y_labels.npy"), y)

    print("Preprocessing complete.")
    print("Saved features shape:", X.shape)
    print("Saved labels shape:", y.shape)


if __name__ == "__main__":
    build_feature_dataset()
