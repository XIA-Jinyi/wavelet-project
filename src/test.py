"""Unified testing — load trained models and output predictions to CSV."""
import argparse
import numpy as np
import torch
import pickle
import csv
from pathlib import Path

from config import FEATURE_DIR, MODEL_DIR, RESULT_DIR, DEVICE, WAVELETS, RATES
from models.wdcnn import StegoCNN
from models.mlp import FeatureMLP
from models.cnn import PureCNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wavelet", default="db4", choices=WAVELETS)
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--n-train", type=int, default=350)
    parser.add_argument("--n-test", type=int, default=150)
    parser.add_argument("--models", nargs="+", default=["wdcnn", "mlp", "svm", "cnn"],
                        choices=["wdcnn", "mlp", "svm", "cnn"])
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.wavelet}_{args.rate:.1f}_{args.n_train}"

    # Load test data
    n_test = args.n_test
    y_test = np.hstack([np.zeros(n_test), np.ones(n_test)])

    for model_name in args.models:
        print(f"\nTesting {model_name} ...")
        save_path = RESULT_DIR / f"{model_name}_{suffix}.csv"

        if model_name == "wdcnn":
            Sc_te = np.load(FEATURE_DIR / f"subbands_cover_test_{suffix}.npy")
            Ss_te = np.load(FEATURE_DIR / f"subbands_stego_test_{suffix}.npy")
            X_test = np.vstack([Sc_te[:n_test], Ss_te[:n_test]])
            Xt = torch.tensor(X_test.astype(np.float32)).to(DEVICE)

            model = StegoCNN(in_channels=4).to(DEVICE)
            ckpt = torch.load(MODEL_DIR / f"wdcnn_{suffix}.pt", map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            with torch.no_grad():
                prob = model(Xt).cpu().numpy().ravel()

        elif model_name == "mlp":
            Fc_te = np.load(FEATURE_DIR / f"feat_cover_test_{suffix}.npy")
            Fs_te = np.load(FEATURE_DIR / f"feat_stego_test_{suffix}.npy")
            X_test = np.vstack([Fc_te[:n_test], Fs_te[:n_test]])

            ckpt = torch.load(MODEL_DIR / f"mlp_{suffix}.pt", map_location="cpu", weights_only=False)
            X_test_s = ckpt["scaler"].transform(X_test)
            Xt = torch.tensor(X_test_s.astype(np.float32)).to(DEVICE)

            model = FeatureMLP(input_dim=X_test_s.shape[1]).to(DEVICE)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            with torch.no_grad():
                prob = model(Xt).cpu().numpy().ravel()

        elif model_name == "svm":
            Fc_te = np.load(FEATURE_DIR / f"feat_cover_test_{suffix}.npy")
            Fs_te = np.load(FEATURE_DIR / f"feat_stego_test_{suffix}.npy")
            X_test = np.vstack([Fc_te[:n_test], Fs_te[:n_test]])

            with open(MODEL_DIR / f"svm_{suffix}.pkl", "rb") as f:
                data = pickle.load(f)
            X_test_s = data["scaler"].transform(X_test)
            prob = data["model"].predict_proba(X_test_s)[:, 1]

        elif model_name == "cnn":
            Rc_te = np.load(FEATURE_DIR / f"raw_cover_test_{args.n_train}.npy")
            Rs_te = np.load(FEATURE_DIR / f"raw_stego_test_{args.rate:.1f}_{args.n_train}.npy")
            X_test = np.vstack([Rc_te[:n_test], Rs_te[:n_test]])

            mean = X_test.mean(axis=(1, 2), keepdims=True)
            std = X_test.std(axis=(1, 2), keepdims=True) + 1e-8
            X_test = (X_test - mean) / std

            Xt = torch.tensor(X_test.astype(np.float32)).unsqueeze(1).to(DEVICE)

            model = PureCNN(in_channels=1).to(DEVICE)
            ckpt = torch.load(MODEL_DIR / f"cnn_{args.rate:.1f}_{args.n_train}.pt",
                              map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            with torch.no_grad():
                prob = model(Xt).cpu().numpy().ravel()
            save_path = RESULT_DIR / f"cnn_{args.rate:.1f}_{args.n_train}.csv"

        pred = (prob > 0.5).astype(int)
        with open(save_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_id", "label", "predicted_prob", "predicted_class"])
            for i in range(len(y_test)):
                writer.writerow([i, int(y_test[i]), float(prob[i]), int(pred[i])])
        print(f"  Saved to {save_path}")


if __name__ == "__main__":
    main()
