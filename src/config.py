"""Shared configuration for the steganalysis project."""
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset"
MODEL_DIR = ROOT / "model"
OUTPUT_DIR = ROOT / "output"
FEATURE_DIR = OUTPUT_DIR / "features"
RESULT_DIR = OUTPUT_DIR / "results"
FIGURE_DIR = OUTPUT_DIR / "figures"

WAVELETS = ["haar", "db4"]
RATES = [0.2, 0.6, 1.0]

# Device selection: cuda > mps > cpu
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# Spearman-confirmed relevant features (12-dim, Level 1 high-frequency subbands)
RELEVANT_FEATURE_NAMES = [
    "LH1_entropy", "LH1_kurt", "LH1_energy", "LH1_var",
    "HL1_entropy", "HL1_kurt", "HL1_energy", "HL1_var",
    "HH1_entropy", "HH1_kurt", "HH1_energy", "HH1_var",
]

# All 60 feature names (for reference)
ALL_FEATURE_NAMES = []
for lv in [3]:
    for stat in ['mean', 'var', 'skew', 'kurt', 'energy', 'entropy']:
        ALL_FEATURE_NAMES.append(f'LL{lv}_{stat}')
for lv in range(3, 0, -1):
    for band in ['LH', 'HL', 'HH']:
        for stat in ['mean', 'var', 'skew', 'kurt', 'energy', 'entropy']:
            ALL_FEATURE_NAMES.append(f'{band}{lv}_{stat}')
