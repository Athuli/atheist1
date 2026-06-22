import numpy as np
from scipy.stats import kurtosis, skew

def extract_features(signal):
    feat = {}
    feat["mean"] = np.mean(signal)
    feat["std"] = np.std(signal)
    feat["rms"] = np.sqrt(np.mean(np.square(signal)))
    feat["peak"] = np.max(np.abs(signal))
    feat["kurtosis"] = kurtosis(signal)
    feat["skewness"] = skew(signal)
    feat["wave_factor"] = feat["rms"] / abs(feat["mean"])
    feat["pulse_factor"] = feat["peak"] / abs(feat["mean"])
    feat["margin_factor"] = feat["peak"] / np.sqrt(feat["std"])
    return feat