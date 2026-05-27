"""LSB embedding supporting arbitrary bit-per-pixel rates."""
import numpy as np


def embed_lsb(img: np.ndarray, rate_bpp: float, seed: int) -> np.ndarray:
    """LSB replacement across up to 3 lowest bit-planes.

    Args:
        img: float64 array in [0, 1].
        rate_bpp: bits per pixel to embed.
        seed: RNG seed for reproducibility.

    Returns:
        Stego image as float64 in [0, 1].
    """
    rng = np.random.default_rng(seed)
    n_pixels = img.size
    total_bits = int(n_pixels * rate_bpp)
    quant = np.round(img * 255).astype(np.uint8).ravel()

    bp0 = min(total_bits, n_pixels)
    bp1 = min(max(0, total_bits - n_pixels), n_pixels)
    bp2 = max(0, total_bits - 2 * n_pixels)

    for bp, n_bits in enumerate([bp0, bp1, bp2]):
        if n_bits <= 0:
            break
        idx = rng.choice(n_pixels, size=n_bits, replace=False)
        bits = rng.integers(0, 2, size=n_bits, dtype=np.uint8)
        mask = ~(1 << bp) & 0xFF
        quant[idx] = (quant[idx] & mask) | (bits << bp)

    return quant.reshape(img.shape).astype(np.float64) / 255.0
