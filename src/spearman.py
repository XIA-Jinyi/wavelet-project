"""Spearman correlation analysis — determines which features are informative."""
import argparse
import numpy as np
from PIL import Image
from scipy import stats
import pywt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from config import DATA_DIR, FIGURE_DIR, ALL_FEATURE_NAMES, RELEVANT_FEATURE_NAMES


def extract_all_60(img, wavelet="db4"):
    """Extract all 60 features from 3-level DWT (for Spearman analysis)."""
    coeffs = pywt.wavedec2(img, wavelet=wavelet, mode="periodization", level=3)
    n = len(coeffs) - 1
    subbands = {f"LL{n}": coeffs[0]}
    for i, (LH, HL, HH) in enumerate(coeffs[1:]):
        lv = n - i
        subbands[f"LH{lv}"] = LH; subbands[f"HL{lv}"] = HL; subbands[f"HH{lv}"] = HH
    ordered = [f"LL{n}"]
    for lv in range(n, 0, -1):
        for band in ["LH", "HL", "HH"]:
            ordered.append(f"{band}{lv}")
    total_energy = sum(np.sum(subbands[nm] ** 2) for nm in ordered)
    feats = []
    for nm in ordered:
        flat = subbands[nm].ravel(); m = np.mean(flat); v = np.var(flat)
        sk = float(stats.skew(flat, bias=False)) if v > 1e-15 else 0.0
        kt = float(stats.kurtosis(flat, fisher=True, bias=False)) if v > 1e-15 else 0.0
        er = float(np.sum(subbands[nm] ** 2)) / total_energy if total_energy > 0 else 0.0
        cnt, _ = np.histogram(flat, bins=64)
        prob = cnt.astype(np.float64) / max(np.sum(cnt), 1); prob = prob[prob > 0]
        ent = float(-np.sum(prob * np.log2(prob))) if len(prob) > 0 else 0.0
        feats.extend([m, v, sk, kt, er, ent])
    return np.array(feats, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=350)
    parser.add_argument("--rate", type=float, default=2.0)
    parser.add_argument("--wavelet", default="db4")
    args = parser.parse_args()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = DATA_DIR / "BOSSbase_1.01"
    files = sorted(data_dir.glob("*.pgm"), key=lambda x: int(x.stem))[:args.n_train * 2]

    print(f"Loading {len(files)} images for Spearman analysis (wavelet={args.wavelet}, rate={args.rate}) ...")
    images = [np.array(Image.open(f), dtype=np.float64) / 255.0 for f in tqdm(files)]

    from embed import embed_lsb
    np.random.seed(42)
    seeds = np.random.randint(0, 2**31, size=len(images))

    # Extract 60-d features for cover and stego
    print("Extracting cover features ...")
    F_cover = np.array([extract_all_60(img, args.wavelet) for img in tqdm(images)])
    print("Extracting stego features ...")
    F_stego = np.array([extract_all_60(embed_lsb(img, args.rate, int(s)), args.wavelet)
                         for img, s in tqdm(zip(images, seeds), total=len(images))])

    X = np.vstack([F_cover, F_stego])
    y = np.hstack([np.zeros(len(images)), np.ones(len(images))])

    # Spearman per feature
    print("Computing Spearman correlations ...")
    rho = np.array([stats.spearmanr(X[:, j], y).correlation for j in tqdm(range(60))])

    # Top features
    top_idx = np.argsort(-np.abs(rho))
    print("\nTop 15 features by Spearman |rho|:")
    for k in range(15):
        j = top_idx[k]
        print(f"  {k+1:2d}. {ALL_FEATURE_NAMES[j]:>20s}  rho={rho[j]:+.4f}")

    # Level breakdown
    for lv_name, lv_range in [("Level 1", slice(24, 60)), ("Level 2", slice(6, 24)), ("Level 3", slice(0, 6))]:
        lv_rho = np.abs(rho[lv_range])
        print(f"  {lv_name}: mean |rho| = {lv_rho.mean():.4f}, max = {lv_rho.max():.4f}")

    # Validate the 12 selected features
    sel_idx = [ALL_FEATURE_NAMES.index(n) for n in RELEVANT_FEATURE_NAMES]
    sel_rho = np.abs(rho[sel_idx])
    unsel_rho = np.abs(rho[[i for i in range(60) if i not in sel_idx]])
    print(f"\n  Selected 12 features: mean |rho| = {sel_rho.mean():.4f}")
    print(f"  Other 48 features:   mean |rho| = {unsel_rho.mean():.4f}")

    # Heatmap
    fig, ax = plt.subplots(figsize=(22, 4))
    rho_mat = rho.reshape(1, -1)
    im = ax.imshow(np.abs(rho_mat), aspect="auto", cmap="YlOrRd", vmin=0, vmax=0.6)
    ax.set_yticks([])
    ax.set_xticks(range(60))
    ax.set_xticklabels(ALL_FEATURE_NAMES, rotation=90, fontsize=5.5)
    ax.set_title(f"Spearman |rho| — {args.wavelet} @ {args.rate:.1f} bpp (N={len(images)*2})")
    plt.colorbar(im, ax=ax, fraction=0.02)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"spearman_heatmap_{args.wavelet}_{args.rate:.1f}.png", dpi=150)
    plt.close()

    # Top-15 bar
    fig, ax = plt.subplots(figsize=(10, 6))
    top15 = top_idx[:15]
    colors = ["#e74c3c" if "HH1" in ALL_FEATURE_NAMES[j] else
              "#3498db" if "HL1" in ALL_FEATURE_NAMES[j] else
              "#2ecc71" if "LH1" in ALL_FEATURE_NAMES[j] else "#95a5a6"
              for j in top15]
    ax.barh(range(15), np.abs(rho[top15]), color=colors, edgecolor="white")
    ax.set_yticks(range(15))
    ax.set_yticklabels([ALL_FEATURE_NAMES[j] for j in top15], fontsize=9)
    ax.set_xlabel("Spearman |rho|")
    ax.set_title(f"Top 15 features — {args.wavelet} @ {args.rate:.1f} bpp")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"spearman_top15_{args.wavelet}_{args.rate:.1f}.png", dpi=150)
    plt.close()

    print(f"\nSaved figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
