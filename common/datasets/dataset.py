# datasets/biopsy_dataset.py
from pathlib import Path
import re
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor


GRADE2IDX = {"0R": 0, "1R": 1, "2R": 2, "3R": 3}
BIN_MAP   = {0: 0, 1: 0, 2: 1, 3: 1}          # 0R/1R → class 0, 2R/3R → class 1
GRADE_RE  = re.compile(r"_(0R|1R[^_/]*|2R|3R)")

def grade_from_path(path: Path) -> int:
    """Return integer label 0-3 extracted from file path."""
    m = GRADE_RE.search(path.as_posix())
    if not m:
        raise ValueError(f"No grade token in {path}")
    return GRADE2IDX[m.group(1)[:2]]           # '1R1AQ' → '1R' → 1


class BiopsyDataset(Dataset):
    """Grayscale biopsy images with on-the-fly resize + ToTensor (binary labels)."""
    def __init__(self, root: str | Path, size: int = 256):
        self.root  = Path(root).expanduser().resolve()
        #self.files = sorted(self.root.rglob("*.tif"))
        # ignore macOS resource-fork files like ._foo.tif
        self.files = sorted(
            p for p in self.root.rglob("*.tif") if not p.name.startswith("._")
        )
        if not self.files:
            raise RuntimeError(f"No .tif files under {self.root}")

        # --- original 4-class mapping (kept for reference) ---
        # self.labels = [grade_from_path(p) for p in self.files]

        #binary mapping
        raw_labels  = [grade_from_path(p) for p in self.files]
        self.labels = [BIN_MAP[l] for l in raw_labels]

        self.tf = Compose([Resize((size, size)), ToTensor()])  # returns [1,H,W] in [0,1]

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("L")
        return self.tf(img), self.labels[idx]


class BiopsyCSVDataset(Dataset):
    """
    CSV must contain at least columns 'image' and 'target'.
    Each row points to an image file and its original 0-3 grade.
    Only binary labels are returned.
    """
    def __init__(
        self,
        csv_file: str | Path,
        root: str | Path | None = None,
        size: int = 256,
    ):
        self.df = pd.read_csv(csv_file)
        if not {"image", "target"}.issubset(self.df.columns):
            raise ValueError("CSV must contain columns 'image' and 'target'.")

        self.root = Path(root).expanduser().resolve() if root else None
        self.paths = [
            (self.root / p if self.root else Path(p)).expanduser().resolve()
            for p in self.df["image"]
        ]
        if not all(p.exists() for p in self.paths):
            missing = [p for p in self.paths if not p.exists()][:3]
            raise RuntimeError(f"Missing image files {missing}")

        # --- original 4-class labels (kept for reference) ---
        # self.labels = self.df["target"].astype(int).tolist()

        # binary mapping
        raw_labels  = self.df["target"].astype(int).tolist()
        self.labels = [BIN_MAP[l] for l in raw_labels]

        self.tf = Compose([Resize((size, size)), ToTensor()])

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("L")
        return self.tf(img), self.labels[idx]

# Quick test
if __name__ == "__main__":
    from torch.utils.data import DataLoader, random_split

    ROOT = "2017"  # adjust

    ds = BiopsyDataset(ROOT, size=256)
    train_ds, val_ds = random_split(
        ds, [int(0.8 * len(ds)), len(ds) - int(0.8 * len(ds))],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    imgs, lbls = next(iter(train_loader))
    print("Batch shape:", imgs.shape)      # expected: [8, 1, 256, 256]
    print("Labels:", lbls.tolist())        # should show only 0 or 1
