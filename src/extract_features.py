"""Batch feature extraction with caching."""
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from config import DATA_DIR, FEATURE_DIR, WAVELETS, RATES
from embed import embed_lsb
from features import extract_subbands, extract_relevant_features


def load_images(file_list):
    return [np.array(Image.open(f), dtype=np.float64) / 255.0 for f in tqdm(file_list, desc="Loading")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wavelet", default="db4", choices=WAVELETS)
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--n-train", type=int, default=350)
    parser.add_argument("--n-test", type=int, default=150)
    parser.add_argument("--raw-only", action="store_true",
                        help="Only cache raw images for PureCNN (skip wavelet extraction)")
    args = parser.parse_args()

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = DATA_DIR / "BOSSbase_1.01"
    if not data_dir.exists():
        raise FileNotFoundError(f"{data_dir} not found. Run 'make data' first.")

    files = sorted(data_dir.glob("*.pgm"), key=lambda x: int(x.stem))
    train_files = files[:args.n_train]
    test_files = files[args.n_train:args.n_train + args.n_test]

    train_imgs = load_images(train_files)
    test_imgs = load_images(test_files)

    suffix = f"{args.wavelet}_{args.rate:.1f}_{args.n_train}"

    # Cache raw cover images for PureCNN (wavelet-independent, rate-independent)
    raw_cover_tr_path = FEATURE_DIR / f"raw_cover_train_{args.n_train}.npy"
    raw_cover_te_path = FEATURE_DIR / f"raw_cover_test_{args.n_train}.npy"
    if not raw_cover_tr_path.exists():
        np.save(raw_cover_tr_path, np.array(train_imgs, dtype=np.float32))
        print(f"Saved {raw_cover_tr_path.name}")
    if not raw_cover_te_path.exists():
        np.save(raw_cover_te_path, np.array(test_imgs, dtype=np.float32))
        print(f"Saved {raw_cover_te_path.name}")

    np.random.seed(42)
    seeds_tr = np.random.randint(0, 2**31, size=len(train_imgs))
    seeds_te = np.random.randint(0, 2**31, size=len(test_imgs))

    # Cache raw stego images for PureCNN (rate-dependent, wavelet-independent)
    raw_stego_tr_path = FEATURE_DIR / f"raw_stego_train_{args.rate:.1f}_{args.n_train}.npy"
    raw_stego_te_path = FEATURE_DIR / f"raw_stego_test_{args.rate:.1f}_{args.n_train}.npy"
    if not raw_stego_tr_path.exists():
        stego_raw_tr = np.array([embed_lsb(img, args.rate, int(s))
                                 for img, s in tqdm(zip(train_imgs, seeds_tr), total=len(train_imgs), desc="Raw stego train")],
                                dtype=np.float32)
        np.save(raw_stego_tr_path, stego_raw_tr)
        print(f"Saved {raw_stego_tr_path.name}")
    if not raw_stego_te_path.exists():
        stego_raw_te = np.array([embed_lsb(img, args.rate, int(s))
                                 for img, s in tqdm(zip(test_imgs, seeds_te), total=len(test_imgs), desc="Raw stego test")],
                                dtype=np.float32)
        np.save(raw_stego_te_path, stego_raw_te)
        print(f"Saved {raw_stego_te_path.name}")

    if args.raw_only:
        print("Raw-only mode: skipping wavelet feature extraction.")
        return

    # Subbands for WDCNN
    print("Extracting subbands (cover) ...")
    Sc_tr = np.array([extract_subbands(img, args.wavelet) for img in tqdm(train_imgs)])
    Sc_te = np.array([extract_subbands(img, args.wavelet) for img in tqdm(test_imgs)])

    print("Extracting subbands (stego) ...")
    Ss_tr = np.array([extract_subbands(embed_lsb(img, args.rate, int(s)), args.wavelet)
                      for img, s in tqdm(zip(train_imgs, seeds_tr), total=len(train_imgs))])
    Ss_te = np.array([extract_subbands(embed_lsb(img, args.rate, int(s)), args.wavelet)
                      for img, s in tqdm(zip(test_imgs, seeds_te), total=len(test_imgs))])

    np.save(FEATURE_DIR / f"subbands_cover_train_{suffix}.npy", Sc_tr)
    np.save(FEATURE_DIR / f"subbands_cover_test_{suffix}.npy", Sc_te)
    np.save(FEATURE_DIR / f"subbands_stego_train_{suffix}.npy", Ss_tr)
    np.save(FEATURE_DIR / f"subbands_stego_test_{suffix}.npy", Ss_te)

    # 12-d features for MLP/SVM
    print("Extracting relevant features (cover) ...")
    Fc_tr = np.array([extract_relevant_features(img, args.wavelet) for img in tqdm(train_imgs)])
    Fc_te = np.array([extract_relevant_features(img, args.wavelet) for img in tqdm(test_imgs)])

    print("Extracting relevant features (stego) ...")
    Fs_tr = np.array([extract_relevant_features(embed_lsb(img, args.rate, int(s)), args.wavelet)
                      for img, s in tqdm(zip(train_imgs, seeds_tr), total=len(train_imgs))])
    Fs_te = np.array([extract_relevant_features(embed_lsb(img, args.rate, int(s)), args.wavelet)
                      for img, s in tqdm(zip(test_imgs, seeds_te), total=len(test_imgs))])

    np.save(FEATURE_DIR / f"feat_cover_train_{suffix}.npy", Fc_tr)
    np.save(FEATURE_DIR / f"feat_cover_test_{suffix}.npy", Fc_te)
    np.save(FEATURE_DIR / f"feat_stego_train_{suffix}.npy", Fs_tr)
    np.save(FEATURE_DIR / f"feat_stego_test_{suffix}.npy", Fs_te)

    print("Done.")


if __name__ == "__main__":
    main()
