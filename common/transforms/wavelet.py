from __future__ import annotations
import numpy as np
import torch
from PIL import Image
import pywt
import torchvision.transforms as T

__all__ = [
    "WaveletHaar", "WaveletDb4",
    "build_transforms_haar", "build_transforms_db4"
]

# ------------------------------------------------------------------ #
# 1) OSNOVNA klasa za 1 razinu SWT-a (bez pod-uzorkovanja)
# ------------------------------------------------------------------ #
class _WaveletBase:
    def __init__(self, wavelet: str = "haar", img_size: int = 256):
        self.wavelet = wavelet
        self.img_size = img_size

    def _dwt_swt(self, arr: np.ndarray):
        """
        Jedna razina *stacionarne* wavelet-transformacije (SWT-2D).
        Vraća 4 koeficijenta istih dimenzija kao ulaz.
        """
        (cA, (cH, cV, cD)) = pywt.swt2(arr, wavelet=self.wavelet, level=1)[0]
        return np.stack([cA, cH, cV, cD], axis=0)  # [4,H,W]

    def __call__(self, img: Image.Image) -> torch.Tensor:
        # 1) grayscale + fiksna rezolucija ---------------------------
        if img.mode != "L":
            img = img.convert("L")
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)

        # 2) u numpy & normaliziraj u [-1,1] ------------------------
        arr = np.array(img, dtype=np.float32) / 255.0 * 2.0 - 1.0

        # 3) SWT -----------------------------------------------------
        coeffs = self._dwt_swt(arr)                 # [4,H,W]

        # 4) standardizacija kanala (μ=0, σ=1) ----------------------
        coeffs = (coeffs - coeffs.mean(axis=(1, 2), keepdims=True)) / \
                 (coeffs.std(axis=(1, 2), keepdims=True) + 1e-6)

        return torch.from_numpy(coeffs).float()     # Tensor [4,H,W]


# ------------------------------------------------------------------ #
# 2) Haar i Db4
# ------------------------------------------------------------------ #
class WaveletHaar(_WaveletBase):
    def __init__(self, img_size: int = 256):
        super().__init__(wavelet="haar", img_size=img_size)


class WaveletDb4(_WaveletBase):
    def __init__(self, img_size: int = 256):
        super().__init__(wavelet="db4", img_size=img_size)


# ------------------------------------------------------------------ #
_LIGHT_AUG = T.RandomChoice([
    T.RandomHorizontalFlip(p=1.0),
    T.RandomVerticalFlip(p=1.0),
    T.RandomRotation(15)
])

def _make_pipeline(core_tf_cls, img_size: int):
    train_tf = T.Compose([
        _LIGHT_AUG,                        
        T.Lambda(core_tf_cls(img_size))     # Wavelet transform
    ])
    val_tf = T.Compose([
        T.Lambda(core_tf_cls(img_size))
    ])
    return train_tf, val_tf


def build_transforms_haar(img_size: int = 256):
    """Pipeline s Haar (`db1`) valićem – 4 kanala, SWT-1 razina."""
    return _make_pipeline(WaveletHaar, img_size)


def build_transforms_db4(img_size: int = 256):
    """Pipeline s Daubechies-4 (`db4`) valićem – finiji detalji."""
    return _make_pipeline(WaveletDb4, img_size)