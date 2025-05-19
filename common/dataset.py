from pathlib import Path
import re
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Compose, Resize, ToTensor

GRADE2IDX = {"0R": 0, "1R": 1, "2R": 2, "3R": 3}
GRADE_RE  = re.compile(r"_(0R|1R[^_/]*|2R|3R)")

def grade_from_path(path: Path) -> int:
    """Return integer label 0-3 extracted from file path."""
    m = GRADE_RE.search(path.as_posix())
    if not m:
        raise ValueError(f"No grade token in {path}")
    return GRADE2IDX[m.group(1)[:2]]       # '1R1AQ' → '1R' → 1

# Dataset
class BiopsyDataset(Dataset):
    """Grayscale biopsy images with on-the-fly resize + ToTensor."""
    def __init__(self, root: str | Path, size: int = 256):
        self.root = Path(root).expanduser().resolve()
        self.files = sorted(self.root.rglob("*.tif"))
        if not self.files:
            raise RuntimeError(f"No .tif files under {self.root}")
        self.labels = [grade_from_path(p) for p in self.files]
        self.tf     = Compose([Resize((size, size)), ToTensor()])  # [1,H,W]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("L")
        return self.tf(img), self.labels[idx]


if __name__ == "__main__":
    ROOT = "2017"     

    ds = BiopsyDataset(ROOT, size=256)
    train_ds, val_ds = random_split(
        ds, [int(0.8 * len(ds)), len(ds) - int(0.8 * len(ds))],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_ds, batch_size=8, shuffle=True, num_workers=0
    )

    imgs, lbls = next(iter(train_loader))
    print("Batch shape:", imgs.shape)       
    print("Labels:", lbls.tolist())          

