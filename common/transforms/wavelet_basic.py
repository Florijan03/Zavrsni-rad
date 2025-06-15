from __future__ import annotations
from typing import Tuple
from pathlib import Path
import numpy as np
from PIL import Image
import pywt
import torchvision.transforms as T

__all__ = ["WaveletTransform", "build_transforms"]

class WaveletTransform:
    """Pretvara 1‑kanalnu sliku u 3‑kanalni tensor (LL, LH, HL)."""
    def __init__(self, wavelet: str = "db2"):
        self.wavelet = wavelet

    def __call__(self, img: Image.Image) -> np.ndarray:  # HWC float32
        if img.mode != "L":
            img = img.convert("L")
        arr = np.array(img, dtype=np.float32)
        LL, (LH, HL, _) = pywt.dwt2(arr, wavelet=self.wavelet)
        comps = [Image.fromarray(c).resize(img.size, Image.BILINEAR) for c in (LL, LH, HL)]
        stacked = np.stack([np.array(c, dtype=np.float32) for c in comps], axis=2)
        return stacked


def build_transforms(img_size: int = 256) -> Tuple[T.Compose, T.Compose]:
    base = [
        T.Lambda(WaveletTransform()),
        T.ToTensor(),
        T.Resize((img_size, img_size)),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ]
    tf = T.Compose(base)
    return tf, tf 