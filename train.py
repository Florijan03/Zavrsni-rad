#!/usr/bin/env python
"""
Train a simple CNN.

Examples: 
python train.py --root /Volumes/T7/zavrsni_rad --subset 1000 --epochs 1 --batch-size 4
"""
from pathlib import Path
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Subset 
from torchvision.transforms import Normalize
from tqdm import tqdm

from collections import defaultdict
import random

# Local imports
from common.datasets.dataset import BiopsyDataset, BiopsyCSVDataset
from common.models.simple_cnn import SimpleCNN


def get_datasets(args):
    tf_norm = Normalize(mean=[0.5], std=[0.5])   # extra transform

    if args.csv is not None:                      # CSV-based dataset
        ds = BiopsyCSVDataset(
            csv_file=args.csv,
            root=args.root,
            size=args.img_size,     
        )
    else:                                         # Folder-based dataset
        ds = BiopsyDataset(
            root=args.root,
            size=args.img_size,                        
        )

    ds.tf.transforms.append(tf_norm)
    return ds


def main():
    parser = argparse.ArgumentParser(
        description="Binary classification training script."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--root", type=Path, help="Root folder containing images")
    group.add_argument("--csv",  type=Path, help="CSV file with image paths")

    parser.add_argument("--img-size",   type=int,   default=256)
    # --binary and --num-classes are not needed any more
    # parser.add_argument("--binary", action="store_true", help="Use 2 classes")
    # parser.add_argument("--num-classes", type=int, default=2, help="Needed only for >2 classes")
    parser.add_argument("--batch-size", type=int,   default=8)
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--subset",     type=int, default=None,
                        help="Use only the first N images (fast test)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Dataset and dataloaders
    ds_full = get_datasets(args)
    NUM_CLASSES = 2                            # Binary classification

    # # Save this for later maybe
    # # apply --subset *before* train/val split
    # if args.subset is not None and args.subset < len(ds_full):
    #     ds_full = Subset(ds_full, range(args.subset))
    #     print(f"Using only the first {args.subset} images")

    if args.subset is not None and args.subset < len(ds_full):
      # ----- build index list per class -----
      buckets = defaultdict(list)          # {label: [indices]}
      for idx, (_, lbl) in enumerate(ds_full):
          buckets[lbl].append(idx)

      # ----- compute how many samples per class -----
      per_class = max(1, args.subset // len(buckets))
      chosen = []
      random.seed(args.seed)
      for lbl, idx_list in buckets.items():
          random.shuffle(idx_list)
          chosen.extend(idx_list[:per_class])

      # If rounding left us short, top-up with random leftovers
      if len(chosen) < args.subset:
          rest = [i for i in range(len(ds_full)) if i not in chosen]
          chosen.extend(random.sample(rest, args.subset - len(chosen)))

      ds_full = Subset(ds_full, chosen)
      print(f"Using a stratified subset of {args.subset} images "
            f"({per_class} per class)")

    n_train = int(0.8 * len(ds_full))            # 80 % training, 20 % validation
    n_val   = len(ds_full) - n_train
    train_ds, val_ds = random_split(
        ds_full, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0
    )
    val_loader   = DataLoader(
        val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # Device selection --------------------------------------------------------
    if torch.backends.mps.is_available():
        device = torch.device("mps")    # Apple Silicon GPU
    elif torch.cuda.is_available():
        device = torch.device("cuda")   # NVIDIA GPU (if any)
    else:
        device = torch.device("cpu")
    print("Device:", device)

    # Model and optimizer -----------------------------------------------------
    model = SimpleCNN(num_classes=NUM_CLASSES).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop -----------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        # ---------- train phase ----------
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)

        acc = correct / total * 100
        print(f"  Train loss: {train_loss/total:.4f}  |  acc: {acc:.2f}%")

        # ---------- validation phase ----------
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += F.cross_entropy(logits, y, reduction="sum").item()
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total   += y.size(0)

        acc = correct / total * 100
        print(f"  Val   loss: {val_loss/total:.4f}  |  acc: {acc:.2f}%")


if __name__ == "__main__":
    main()
