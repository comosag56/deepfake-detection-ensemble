"""
train_full.py - retrain a model on ALL 7,000 labelled photos (no photos held back).

Same recipe as train_model.py (same damage augmentation, loss, optimiser, learning-rate schedule),
so it needs train_model.py in the same folder. The differences:
  * no exam photos and no scoring: every labelled photo is used for learning, so a retrained model
    CANNOT be tested afterwards (it has seen everything). We trust the settings found earlier.
  * --epochs is the length of the learning-rate schedule (use the same number as the validated run) and
    --stop_after is the epoch that was best in that run: the script runs the identical schedule and saves
    the weights at that epoch as final.pt.

Settings from your validated runs:
    efficientnet_b3 (the "b3_aug" run) : --epochs 15 --stop_after 8      training loss at epoch 8 was about 0.39
    convnext_tiny                      : --epochs 12 --stop_after 5      training loss at epoch 5 was about 0.45
    swin_t (the 12-epoch run)          : --epochs 12 --stop_after 10     training loss at epoch 10 was about 0.44

Quick test (1 epoch on 400 photos):
    python train_full.py --data_root hackathon_dataset --model efficientnet_b3 --epochs 15 --stop_after 1 --limit 400 --out_dir runs/quick_full
Real runs:
    python train_full.py --data_root hackathon_dataset --model efficientnet_b3 --epochs 15 --stop_after 8
    python train_full.py --data_root hackathon_dataset --model convnext_tiny --epochs 12 --stop_after 5
    python train_full.py --data_root hackathon_dataset --model swin_t --epochs 12 --stop_after 10

Output (default runs/full_<model>): final.pt (use this in package_models.py), last.pt (for --resume), summary.json.
If interrupted, repeat the same command with --resume.
"""
import argparse
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_model import FIXED_224, MODEL_DEFAULTS, TrainSet, build_model, find_images, seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--model", required=True, choices=list(MODEL_DEFAULTS))
    ap.add_argument("--epochs", type=int, required=True, help="length of the learning-rate schedule")
    ap.add_argument("--stop_after", type=int, default=0, help="stop and save after this many epochs (0 = run all)")
    ap.add_argument("--out_dir", default=None, help="default: runs/full_<model>")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--smoothing", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="use only this many photos (quick tests)")
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    lr_default, bs_default = MODEL_DEFAULTS[args.model]
    args.lr = args.lr or lr_default
    args.batch_size = args.batch_size or bs_default
    args.out_dir = args.out_dir or f"runs/full_{args.model}"
    stop = args.stop_after or args.epochs
    if stop > args.epochs:
        raise SystemExit("--stop_after cannot be larger than --epochs")
    if args.model in FIXED_224 and args.img_size != 224:
        raise SystemExit(f"{args.model} only works at --img_size 224")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"model {args.model}   lr {args.lr}   batch {args.batch_size}   schedule {args.epochs} epochs, "
          f"stopping after {stop}   -> {out}")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if use_amp else "  <-- no GPU found!"))

    # ---- ALL labelled photos ---------------------------------------------------
    df = pd.read_csv(Path(args.data_root) / "train_metadata.csv")
    df["image_id"] = df["image_id"].astype(str).str.strip()
    files = find_images(args.data_root)
    missing = [i for i in df["image_id"] if i not in files]
    if missing:
        raise SystemExit(f"{len(missing)} images from train_metadata.csv not found, e.g. {missing[:3]}")
    if args.limit:
        df = df.sample(n=args.limit, random_state=args.seed)
    paths = [files[i] for i in df["image_id"]]
    labels = df["label"].tolist()
    print(f"training on {len(paths)} photos (no photos held back)")

    g = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(TrainSet(paths, labels, args.img_size), batch_size=args.batch_size, shuffle=True,
                        drop_last=True, num_workers=args.num_workers, pin_memory=use_amp,
                        persistent_workers=args.num_workers > 0, generator=g)

    model = build_model(args.model, not args.no_pretrained, args.dropout).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    total_steps = args.epochs * len(loader)
    warm = max(1, len(loader))

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        prog = (step - warm) / max(1, total_steps - warm)
        return max(0.01, 0.5 * (1 + math.cos(math.pi * prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    start = 0
    if args.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start = ck["epoch"] + 1
        print(f"resumed: continuing from epoch {start + 1}")

    eps = args.smoothing
    last_loss = None
    for epoch in range(start, stop):
        t0 = time.time()
        model.train()
        total, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            target = y * (1 - 2 * eps) + eps
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x).squeeze(1)
            loss = loss_fn(logits.float(), target)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(y)
            n += len(y)
        last_loss = total / max(1, n)
        print(f"epoch {epoch + 1}/{stop} (schedule length {args.epochs})  loss {last_loss:.4f}  ({time.time() - t0:.0f}s)")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch}, out / "last.pt")

    torch.save(model.state_dict(), out / "final.pt")
    summary = {"model": args.model, "trained_on_photos": len(paths), "schedule_epochs": args.epochs,
               "stopped_after": stop, "final_train_loss": None if last_loss is None else round(last_loss, 4),
               "params_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
               "weights_mb": round((out / "final.pt").stat().st_size / 1e6, 1), "args": vars(args)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {out / 'final.pt'}  ({summary['weights_mb']} MB). Use it in package_models.py.")


if __name__ == "__main__":
    main()
