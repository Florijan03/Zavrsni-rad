# data_loading.py
from pathlib import Path
from PIL import Image
import re, torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

GRADE2IDX = {"0R": 0, "1R": 1, "2R": 2, "3R": 3}

def parse_grade(path: Path) -> int:
    m = re.search(r'_(0R|1R[^_/]*|2R|3R)', path.as_posix())
    if m:
        return GRADE2IDX[m.group(1)[:2]]   # '1R1AQ' -> '1R'
    raise ValueError(f"Not annotated {path}")

class XPCIBiopsyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir).expanduser().resolve()
        if not self.root_dir.exists():
            raise FileNotFoundError(f"{self.root_dir} does not exists")
        self.transform = transform
        self.samples = [(p, parse_grade(p))
                        for p in self.root_dir.rglob("*.tif")
                        if p.is_file()]

        if not self.samples:
            raise RuntimeError(f"There is no .tif in {self.root_dir}")

    def __len__(self):  return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")          # grayscale
        img = self.transform(img) if self.transform else transforms.ToTensor()(img)
        return img, label

# -------------------- TEST --------------------
if __name__ == "__main__":
    ROOT = Path(__file__).parent / "2017"           

    # print("Looking in:", ROOT.resolve())
    # print("Example of files:", list(ROOT.rglob("*.tif"))[:5])

    tfm = transforms.Compose([
        transforms.Resize((256, 256)),              
        transforms.ToTensor()
    ])

    ds = XPCIBiopsyDataset(ROOT, transform=tfm)
    train_ds, val_ds = random_split(ds, [int(0.8*len(ds)), len(ds)-int(0.8*len(ds))])

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    imgs, lbls = next(iter(train_loader))
    print("Batch:", imgs.shape, "Labels:", lbls.tolist())
