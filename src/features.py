"""Wavelet-domain feature extraction for steganalysis."""
import numpy as np
import pywt
from scipy import stats


def extract_subbands(img: np.ndarray, wavelet: str = "db4") -> np.ndarray:
    """1-level DWT → 4 standardised subbands (4, H/2, W/2)."""
    coeffs = pywt.wavedec2(img, wavelet=wavelet, mode="periodization", level=1)
    LL, (LH, HL, HH) = coeffs[0], coeffs[1]
    result = []
    for band in [LL, LH, HL, HH]:
        b = (band - band.mean()) / (band.std() + 1e-8)
        result.append(b.astype(np.float32))
    return np.stack(result)


def extract_relevant_features(img: np.ndarray, wavelet: str = "db4") -> np.ndarray:
    """Extract the 12 Spearman-confirmed features from Level 1 subbands.

    Features: LH1/HL1/HH1 × {entropy, kurtosis, energy, variance}
    """
    coeffs = pywt.wavedec2(img, wavelet=wavelet, mode="periodization", level=1)
    _, (LH, HL, HH) = coeffs[0], coeffs[1]
    subbands = {"LH1": LH, "HL1": HL, "HH1": HH}

    features = []
    for _, band in subbands.items():
        flat = band.ravel()
        v = float(np.var(flat))
        kt = float(stats.kurtosis(flat, fisher=True, bias=False)) if v > 1e-15 else 0.0
        e = float(np.sum(band ** 2))
        counts, _ = np.histogram(flat, bins=64)
        prob = counts.astype(np.float64) / max(np.sum(counts), 1)
        prob = prob[prob > 0]
        ent = float(-np.sum(prob * np.log2(prob))) if len(prob) > 0 else 0.0
        features.extend([ent, kt, e, v])
    return np.array(features, dtype=np.float32)
