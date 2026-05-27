"""SVM training script — uses Spearman-selected 12-d features."""
import argparse
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from pathlib import Path

from config import FEATURE_DIR, MODEL_DIR, WAVELETS, RATES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wavelet", default="db4", choices=WAVELETS)
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--n-train", type=int, default=350)
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.wavelet}_{args.rate:.1f}_{args.n_train}"

    Fc_tr = np.load(FEATURE_DIR / f"feat_cover_train_{suffix}.npy")
    Fs_tr = np.load(FEATURE_DIR / f"feat_stego_train_{suffix}.npy")
    X = np.vstack([Fc_tr, Fs_tr])
    y = np.hstack([np.zeros(len(Fc_tr)), np.ones(len(Fs_tr))])

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    print(f"Training SVM (linear, C=1.0): wavelet={args.wavelet}, rate={args.rate}, n={len(X)}")
    svm = SVC(kernel="linear", C=1.0, probability=True, random_state=42)
    svm.fit(X_s, y)

    save_path = MODEL_DIR / f"svm_{suffix}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump({"model": svm, "scaler": scaler}, f)
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
