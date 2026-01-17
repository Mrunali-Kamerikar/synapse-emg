import numpy as np
import pandas as pd

def normalize_signal(signal):
    """
    Z-score normalization per channel
    signal: (T, 8)
    """
    mean = np.mean(signal, axis=0)
    std = np.std(signal, axis=0) + 1e-8
    return (signal - mean) / std


def extract_features(signal):
    """
    Extract time-domain EMG features
    signal: (T, 8)
    returns: (48,)
    """
    features = []

    for ch in range(signal.shape[1]):
        channel = signal[:, ch]

        # Existing features
        mav = np.mean(np.abs(channel))
        rms = np.sqrt(np.mean(channel ** 2))
        var = np.var(channel)
        max_val = np.max(np.abs(channel))

        # NEW features
        wl = np.sum(np.abs(np.diff(channel)))  # Waveform Length

        zc = np.sum(
            ((channel[:-1] * channel[1:]) < 0).astype(int)
        )  # Zero Crossings

        features.extend([mav, rms, var, max_val, wl, zc])

    return np.array(features)



def preprocess_csv(csv_path):
    """
    Full preprocessing for one CSV file
    """
    df = pd.read_csv(csv_path)
    signal = df.values  # (T, 8)

    signal = normalize_signal(signal)
    features = extract_features(signal)

    return features
