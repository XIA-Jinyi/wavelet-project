"""Data preparation: download BOSSBase, extract, and create train/test splits."""
import argparse
import os
import shutil
import zipfile
import urllib.request
import sys
from pathlib import Path

from config import DATA_DIR

BOSS_URL = "https://dde.binghamton.edu/download/ImageDB/BOSSbase_1.01.zip"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=7000)
    parser.add_argument("--n-test", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target_dir = DATA_DIR / "BOSSbase_1.01"
    zip_path = DATA_DIR / "BOSSbase_1.01.zip"

    # Download if needed
    if not target_dir.exists() and not zip_path.exists():
        print(f"Downloading BOSSBase from {BOSS_URL} ...")
        urllib.request.urlretrieve(BOSS_URL, zip_path)
        print("Download complete.")

    # Extract if needed
    if zip_path.exists() and not target_dir.exists():
        print(f"Extracting {zip_path} ...")
        target_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(target_dir)
        # If zip extracted into a subdirectory, move files up
        contents = list(target_dir.iterdir())
        if len(contents) == 1 and contents[0].is_dir():
            inner = contents[0]
            for f in inner.iterdir():
                shutil.move(str(f), str(target_dir / f.name))
            inner.rmdir()
        zip_path.unlink()  # remove zip after extraction
        print("Extraction complete.")

    if not target_dir.exists():
        print(f"Error: {target_dir} not found. Please place BOSSBase images there manually.")
        sys.exit(1)

    # Create train/test file lists
    files = sorted(target_dir.glob("*.pgm"), key=lambda x: int(x.stem))
    print(f"Found {len(files)} images.")
    import numpy as np
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(files))
    train_files = [files[i] for i in indices[:args.n_train]]
    test_files = [files[i] for i in indices[args.n_train:args.n_train + args.n_test]]

    with open(DATA_DIR / "train_files.txt", "w") as f:
        f.write("\n".join(str(p) for p in train_files))
    with open(DATA_DIR / "test_files.txt", "w") as f:
        f.write("\n".join(str(p) for p in test_files))

    print(f"Train: {len(train_files)}, Test: {len(test_files)}")
    print("Data preparation done.")


if __name__ == "__main__":
    main()
