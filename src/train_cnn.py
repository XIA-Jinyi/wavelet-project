"""Pure CNN training script -- operates on raw grayscale images."""
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from pathlib import Path
from tqdm import tqdm

from config import FEATURE_DIR, MODEL_DIR, DEVICE, RATES
from models.cnn import PureCNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=float, default=0.6)
    parser.add_argument("--n-train", type=int, default=350)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-split", type=float, default=0.2)
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.rate:.1f}_{args.n_train}"

    # Load raw images
    Rc_tr = np.load(FEATURE_DIR / f"raw_cover_train_{args.n_train}.npy")
    Rs_tr = np.load(FEATURE_DIR / f"raw_stego_train_{args.rate:.1f}_{args.n_train}.npy")
    X = np.vstack([Rc_tr, Rs_tr])
    y = np.hstack([np.zeros(len(Rc_tr)), np.ones(len(Rs_tr))])

    # Per-image standardisation
    mean = X.mean(axis=(1, 2), keepdims=True)
    std = X.std(axis=(1, 2), keepdims=True) + 1e-8
    X = (X - mean) / std
    X = X.astype(np.float32)

    # Train/val split
    rng = np.random.default_rng(99)
    idx = rng.permutation(len(X))
    n_val = int(len(idx) * args.val_split)
    tr_idx = idx[n_val:]
    val_idx = idx[:n_val]

    Xt = torch.tensor(X[tr_idx]).unsqueeze(1).to(DEVICE)
    yt = torch.tensor(y[tr_idx].astype(np.float32)).unsqueeze(1).to(DEVICE)
    Xv = torch.tensor(X[val_idx]).unsqueeze(1).to(DEVICE)
    yv = torch.tensor(y[val_idx].astype(np.float32)).unsqueeze(1).to(DEVICE)

    train_dl = DataLoader(TensorDataset(Xt, yt), batch_size=args.batch_size, shuffle=True)

    model = PureCNN(in_channels=1).to(DEVICE)
    print(f"PureCNN params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {DEVICE}, Rate: {args.rate}, Train: {len(tr_idx)}, Val: {len(val_idx)}")

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_auc, best_state, best_epoch = 0, None, 0

    pbar = tqdm(range(args.epochs), desc="Training")
    for ep in pbar:
        model.train()
        for xb, yb in train_dl:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            vp_list = []
            chunk = (len(Xv) + 3) // 4
            for i in range(0, len(Xv), chunk):
                vp_list.append(model(Xv[i:i+chunk]).cpu().numpy().ravel())
            vp = np.concatenate(vp_list)
            auc = roc_auc_score(yv.cpu().numpy().ravel(), vp)
        if auc > best_auc:
            best_auc, best_epoch = auc, ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        pbar.set_postfix({"auc": f"{auc:.4f}", "best": f"{best_auc:.4f} @ {best_epoch+1}"})

    model.load_state_dict(best_state)
    save_path = MODEL_DIR / f"cnn_{suffix}.pt"
    torch.save({"state_dict": best_state, "best_auc": best_auc, "best_epoch": best_epoch,
                "rate": args.rate, "n_train": args.n_train}, save_path)
    print(f"Saved to {save_path}  (best val AUC={best_auc:.4f} @ epoch {best_epoch+1})")


if __name__ == "__main__":
    main()
