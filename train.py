import argparse, time
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from common.datasets.dataset import (
    BiopsyFolderDataset, BiopsyCSVDataset)
from common.models.resnet_wavelet import WaveletResNet
from common.transforms.wavelet import build_transforms_db4 as build_transforms

from torchmetrics.classification import (
    BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score)

# ────────────────────────────────────────────────────────────────────────────
def verify_dataset(ds):
    from collections import Counter
    print("[VERIFY] Samples per class:", dict(Counter(ds.labels)))

# ────────────────────────────────────────────────────────────────────────────
def run_epoch(model, loader, crit, opt, metrics, device,
              train: bool = True, log_int: int = 20):

    for m in metrics.values():
        m.reset()                                 

    model.train() if train else model.eval()
    tot_loss = tot_ok = 0
    bar = tqdm(loader, desc="train" if train else "val", ncols=100)

    with torch.set_grad_enabled(train):
        for i, (x, y) in enumerate(bar):
            x, y = x.to(device), y.to(device)

            if train:
                opt.zero_grad()

            out = model(x)                    
            loss = crit(out, y)

            if train:
                loss.backward()
                opt.step()

            tot_loss += loss.item() * x.size(0)
            tot_ok   += (out.argmax(1) == y).sum().item()

            # --- update torchmetrics --------------------------------------
            preds = out.softmax(1)[:, 1]            # p(klasa=1)
            for m in metrics.values():
                m.update(preds, y)

            if i % log_int == 0:
                bar.set_postfix(loss=f"{loss.item():.3f}",
                                acc=f"{tot_ok/((i+1)*x.size(0)):.2f}")

    # --- epoch average -----------------------------------------------------
    epoch_loss = tot_loss / len(loader.dataset)
    epoch_acc  = tot_ok   / len(loader.dataset)
    ep_metrics = {name: float(m.compute()) for name, m in metrics.items()}
    return epoch_loss, epoch_acc, ep_metrics   

# ────────────────────────────────────────────────────────────────────────────
def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--train-csv"); p.add_argument("--test-csv")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch",  type=int, default=8)
    p.add_argument("--lr",     type=float, default=1e-3)
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--log-int", type=int, default=20)
    return p.parse_args()

# ────────────────────────────────────────────────────────────────────────────
def main():
    args   = parse()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    train_t, val_t = build_transforms(args.img_size)

    # ------ DATA -----------------------------------------------------------
    if args.train_csv and args.test_csv:
        ds_train = BiopsyCSVDataset(args.train_csv, root=args.root,
                                    size=args.img_size, transform=train_t)
        ds_val   = BiopsyCSVDataset(args.test_csv,  root=args.root,
                                    size=args.img_size, transform=val_t)

    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True,
                          num_workers=2, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=args.batch, shuffle=False,
                          num_workers=2, pin_memory=True)

    # ------ MODEL ----------------------------------------------------------
    model = WaveletResNet().to(device)
    opt   = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit  = nn.CrossEntropyLoss()

    # torchmetrics instance for  every epoch
    def make_metrics():                
        return {
            'acc' : BinaryAccuracy().to(device),
            'rec' : BinaryRecall().to(device),
            'prec': BinaryPrecision().to(device),
            'f1'  : BinaryF1Score().to(device),
        }

    # ------ LOOP -----------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc, tr_m = run_epoch(
            model, dl_train, crit, opt, make_metrics(), device,
            train=True,  log_int=args.log_int)

        vl_loss, vl_acc, vl_m = run_epoch(
            model, dl_val, crit, opt, make_metrics(), device,
            train=False, log_int=args.log_int)

        scheduler.step()

        print(f"Epoch {epoch}/{args.epochs}"
              f" | Train {tr_loss:.4f}/{tr_acc*100:.1f}%"
              f" (F1 {tr_m['f1']:.2f})"
              f" | Val {vl_loss:.4f}/{vl_acc*100:.1f}%"
              f" (F1 {vl_m['f1']:.2f}  Rec {vl_m['rec']:.2f}  Prec {vl_m['prec']:.2f})"
              f" | {time.time()-t0:.1f}s")

    # ------ SAVE -----------------------------------------------------------
    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "models/wavelet_resnet18.pth")
    print("[INFO] Model saved → models/wavelet_resnet18.pth")

# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()