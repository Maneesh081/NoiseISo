"""Training loop for the enhancement U-Net.

Each "step" does:
  1. pick a batch of (noisy picture, clean picture) pairs
  2. predict the mask for the noisy picture
  3. enhance = noisy picture * predicted mask
  4. loss = how far enhanced is from the clean picture (L1)
  5. backprop: walk a tiny step downhill on the loss to update the weights

We alternate between the training set and a validation set. The validation set is a
smaller group of speakers the model has NEVER seen. We watch the loss on that group:
when it stops improving even though the training loss keeps dropping, the model has
started to *memorise* its training data instead of *understanding* the task -- so we
stop early and keep the best checkpoint (the version that generalised the best).
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from dataset import EnhancementDataset, make_loader
from features import Spectrogram
from model import UNet, count_parameters
from utils import set_seed


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"[train] device: {device}")

    # The dataset computes STFTs inside DataLoader worker processes, so it must
    # stay on CPU: inheriting a CUDA tensor into forked workers is unsafe.
    stft = Spectrogram()

    train_ds = EnhancementDataset(
        clean_dir=args.clean_train, noisy_dir=args.noisy_train,
        stft=stft, sr=args.sr, chunk_seconds=args.chunk_seconds, seed=args.seed,
    )
    val_ds = EnhancementDataset(
        clean_dir=args.clean_val, noisy_dir=args.noisy_val,
        stft=stft, sr=args.sr, chunk_seconds=args.chunk_seconds, seed=args.seed + 1,
    )
    train_loader = make_loader(train_ds, args.batch_size, shuffle=True,
                               num_workers=args.workers)
    val_loader = make_loader(val_ds, args.batch_size, shuffle=False,
                             num_workers=args.workers)
    print(f"[train] clips -> train {len(train_ds)} / val {len(val_ds)}")

    model = UNet(base=args.base, depth=args.depth).to(device)
    print(f"[train] U-Net params: {count_parameters(model) / 1e6:.2f} M")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.L1Loss()
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp))

    writer = SummaryWriter(log_dir=os.path.join(args.out_dir, "tb"))
    os.makedirs(os.path.join(args.out_dir, "checkpoints"), exist_ok=True)

    best_val = float("inf")
    wait = 0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n = 0
        t0 = time.time()
        for noisy, clean, scale in train_loader:
            noisy, clean = noisy.unsqueeze(1).to(device), clean.unsqueeze(1).to(device)
            optimizer.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16, enabled=bool(args.amp)):
                mask = model(noisy)
                enhanced = mask * noisy
                loss = criterion(enhanced, clean)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * noisy.size(0)
            n += noisy.size(0)

        scheduler.step()
        train_loss /= max(n, 1)

        model.eval()
        val_loss = 0.0
        vn = 0
        with torch.no_grad():
            for noisy, clean, scale in val_loader:
                noisy, clean = noisy.unsqueeze(1).to(device), clean.unsqueeze(1).to(device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=bool(args.amp)):
                    mask = model(noisy)
                    enhanced = mask * noisy
                    vloss = criterion(enhanced, clean)
                val_loss += vloss.item() * noisy.size(0)
                vn += noisy.size(0)
        val_loss /= max(vn, 1)

        lr = scheduler.get_last_lr()[0]
        print(f"[epoch {epoch:03d}] train {train_loss:.4f} | val {val_loss:.4f} "
              f"| lr {lr:.2e} | {time.time() - t0:.1f}s")
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("lr", lr, epoch)

        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            torch.save(model.state_dict(),
                       os.path.join(args.out_dir, "checkpoints", "best.pt"))
            with open(os.path.join(args.out_dir, "checkpoints", "best_meta.json"), "w") as f:
                json.dump({
                    "config": vars(args),
                    "val_loss": best_val,
                }, f, indent=2, default=str)
            print(f"    -> saved new best (val {best_val:.4f})")
        else:
            wait += 1
            if wait >= args.patience:
                print(f"[train] early stopping at epoch {epoch} (best val {best_val:.4f})")
                break

    print(f"[train] done in {(time.time() - t_start) / 60:.1f} min, best val {best_val:.4f}")
    writer.close()

    config = vars(args)
    config["best_val"] = best_val
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="../data")
    p.add_argument("--out", default="../out")
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--chunk_seconds", type=float, default=3.0)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--base", type=int, default=16)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--max_norm", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clean_val", default=None,
                   help="clean validation dir (default: held-out split, else test set)")
    p.add_argument("--noisy_val", default=None,
                   help="noisy validation dir (default: held-out split, else test set)")
    args = p.parse_args()

    args.clean_train = os.path.join(args.data, "clean_trainset_28spk_wav")
    args.noisy_train = os.path.join(args.data, "noisy_trainset_28spk_wav")

    # Validation: prefer the speaker hold-out, otherwise fall back to the test set.
    val_clean = os.path.join(args.data, "clean_val_wav")
    val_noisy = os.path.join(args.data, "noisy_val_wav")
    if args.clean_val is None:
        args.clean_val = val_clean if os.path.isdir(val_clean) else \
            os.path.join(args.data, "clean_testset_wav")
    if args.noisy_val is None:
        args.noisy_val = val_noisy if os.path.isdir(val_noisy) else \
            os.path.join(args.data, "noisy_testset_wav")
    args.out_dir = args.out

    train(args)
