from __future__ import annotations
import numpy as np, torch, pywt
from PIL import Image
import torchvision.transforms as T

__all__ = ["build_transforms_db4", "build_transforms_haar"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ------------------------------------------------------------------ #
class _WaveletBase:
    """**3** channels (cA, cH, cV)."""
    def __init__(self, wavelet="haar", img_size=256):
        self.wavelet  = wavelet
        self.img_size = img_size

    def __call__(self, img: Image.Image) -> torch.Tensor:
        # 1) grayscale + resize
        if img.mode != "L":
            img = img.convert("L")
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)

        # 2) u [-1,1]  numpy
        arr = np.array(img, np.float32) / 255.0 * 2.0 - 1.0

        # 3) SWT → uzimamo samo cA,cH,cV
        cA,(cH,cV,_) = pywt.swt2(arr, wavelet=self.wavelet, level=1)[0]
        coeffs = np.stack([cA, cH, cV], axis=0)          # [3,H,W]

        # 4) standardizacija po kanalu
        coeffs = (coeffs - coeffs.mean((1,2), keepdims=True)) / (
                 coeffs.std((1,2), keepdims=True) + 1e-6)

        return torch.from_numpy(coeffs).float()

class _WaveletHaar(_WaveletBase):
    def __init__(self, img_size=256): super().__init__("haar", img_size)

class _WaveletDb4(_WaveletBase):
    def __init__(self, img_size=256): super().__init__("db4", img_size)

# ------------------------------------------------------------------ #
_AUG = T.Compose([
    T.RandomResizedCrop(256, scale=(0.8,1.0)),
    T.RandomPerspective(0.2, p=0.5),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(15),
])

def _pipeline(core_cls, img_size):
    core = core_cls(img_size)
    train_tf = T.Compose([_AUG,
                          T.Lambda(core),
                          T.RandomErasing(p=0.5, scale=(0.02,0.33), ratio=(0.3,3.3)),
                          T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    val_tf   = T.Compose([T.Lambda(core),
                          T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return train_tf, val_tf

def build_transforms_haar(img_size=256):
    return _pipeline(_WaveletHaar, img_size)

def build_transforms_db4(img_size=256):
    return _pipeline(_WaveletDb4, img_size)
