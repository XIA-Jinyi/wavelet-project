"""MLP training script — uses Spearman-selected 12-d features."""
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from pathlib import Path

from config import FEATURE_DIR, MODEL_DIR, DEVICE, WAVELETS, RATES
from models.mlp import FeatureMLP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wavelet", default="db4", choices=WAVELETS)
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--n-train", type=int, default=350)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-split", type=float, default=0.2)
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.wavelet}_{args.rate:.1f}_{args.n_train}"

    Fc_tr = np.load(FEATURE_DIR / f"feat_cover_train_{suffix}.npy")
    Fs_tr = np.load(FEATURE_DIR / f"feat_stego_train_{suffix}.npy")
    X = np.vstack([Fc_tr, Fs_tr])
    y = np.hstack([np.zeros(len(Fc_tr)), np.ones(len(Fs_tr))])

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    rng = np.random.default_rng(99)
    idx = rng.permutation(len(X))
    n_val = int(len(idx) * args.val_split)
    tr_idx = idx[n_val:]; val_idx = idx[:n_val]

    Xt = torch.tensor(X[tr_idx].astype(np.float32)).to(DEVICE)
    yt = torch.tensor(y[tr_idx].astype(np.float32)).unsqueeze(1).to(DEVICE)
    Xv = torch.tensor(X[val_idx].astype(np.float32)).to(DEVICE)
    yv = torch.tensor(y[val_idx].astype(np.float32)).unsqueeze(1).to(DEVICE)

    train_dl = DataLoader(TensorDataset(Xt, yt), batch_size=args.batch_size, shuffle=True)

    model = FeatureMLP(input_dim=X.shape[1]).to(DEVICE)
    print(f"MLP params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {DEVICE}, Wavelet: {args.wavelet}, Rate: {args.rate}, Train: {len(tr_idx)}, Val: {len(val_idx)}")

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_auc, best_state, best_epoch = 0, None, 0

    for ep in range(args.epochs):
        model.train()
        for xb, yb in train_dl:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            vp = model(Xv).cpu().numpy().ravel()
            auc = roc_auc_score(yv.cpu().numpy().ravel(), vp)
        if auc > best_auc:
            best_auc, best_epoch = auc, ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (ep + 1) % 20 == 0:
            print(f"  epoch {ep+1:3d}: val AUC={auc:.4f} (best={best_auc:.4f} @ ep {best_epoch+1})")

    model.load_state_dict(best_state)
    save_path = MODEL_DIR / f"mlp_{suffix}.pt"
    torch.save({"state_dict": best_state, "scaler": scaler, "best_auc": best_auc,
                "best_epoch": best_epoch,
                "wavelet": args.wavelet, "rate": args.rate, "n_train": args.n_train}, save_path)
    print(f"Saved to {save_path}  (best val AUC={best_auc:.4f} @ epoch {best_epoch+1})")


if __name__ == "__main__":
    main()
