import os
import numpy as np
from preprocess import preprocess_csv

def get_label_from_filename(filename):
    """
    gesture00_trial01.csv -> label 0
    """
    gesture_str = filename.split('_')[0]  # gesture00
    label = int(gesture_str.replace('gesture', ''))
    return label


def load_dataset(root_dir):
    """
    root_dir: path to Synapse_Dataset
    returns: X, y
    """
    X = []
    y = []

    for session in os.listdir(root_dir):
        session_path = os.path.join(root_dir, session)
        if not os.path.isdir(session_path):
            continue

        for subject in os.listdir(session_path):
            subject_path = os.path.join(session_path, subject)
            if not os.path.isdir(subject_path):
                continue

            for file in os.listdir(subject_path):
                if not file.endswith('.csv'):
                    continue

                file_path = os.path.join(subject_path, file)

                features = preprocess_csv(file_path)
                label = get_label_from_filename(file)

                X.append(features)
                y.append(label)

    return np.array(X), np.array(y)
