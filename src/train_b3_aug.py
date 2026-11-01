"""
train_b3_aug.py - EfficientNet-B3 with 7 EXTRA kinds of damage in training and validation.
(Same as train_b3.py, which is left untouched, plus the damage types seen in the 'confidently wrong'
photos: grey, inverted colours, extreme contrast, vertical streaks/ghosting, white snow,
coloured patches, sketch look. Also uses ONE validation loader for all damages, so far fewer
helper processes start up on Windows.)

Run (from the folder that holds this file, venv active):
    python train_b3_aug.py --data_root "C:\\path\\to\\hackathon_dataset"

If you see 'CUDA out of memory', add --batch_size 16. To make epochs faster, add --damaged_eval_n 700
(then the damaged scores use 700 photos and are no longer exactly comparable with the old runs).

Quick test first (a few hundred images, 1 epoch, finishes in a few minutes):
    python train_b3_aug.py --data_root "C:\\path\\to\\hackathon_dataset" --limit 400 --epochs 1 --out_dir runs/quick

If a run is interrupted, continue it with the same command plus --resume.

Needs:  pip install scikit-learn pandas   (torch + torchvision are already installed)

What it writes into --out_dir (default runs/b3_aug - separate from runs/b0 and runs/b3):
    splits.csv            which images are train / val (copied from runs/b3/splits.csv if that exists,
                          so every model is tested on exactly the same exam photos)
    training_log.csv      now also has auc_damaged_orig7_mean = mean over the ORIGINAL 7 damages,
                          which you can compare directly with the B0 / B3 runs
    training_log.csv      one row per epoch (loss, AUC clean, AUC per damage, time)
    best.pt               best weights (state_dict) - this is what you submit
    last.pt               latest checkpoint, used by --resume
    val_predictions.csv   predictions on the validation images
    summary.json          final metrics + model size / speed / GPU memory
The test folder is never opened by this script.
"""
import argparse
import io
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import models, transforms

MEAN = (0.485, 0.456, 0.406)   # ImageNet statistics (the pretrained model expects these)
STD = (0.229, 0.224, 0.225)


# =============================================================================
# DAMAGE FUNCTIONS  (copy the kinds of damage seen in the sample photos)
# Each takes (PIL image, numpy random generator, severity s in [0,1]) -> PIL image
# =============================================================================
def lerp(a, b, s):
    return a + (b - a) * s


def d_noise(img, rng, s):
    sigma = lerp(8, 40, s)
    arr = np.asarray(img, dtype=np.float32)
    arr = arr + rng.normal(0.0, sigma, arr.shape).astype(np.float32)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def d_saltpepper(img, rng, s):
    amount = lerp(0.01, 0.08, s)
    arr = np.array(img)
    m = rng.random(arr.shape[:2])
    arr[m < amount / 2] = 0
    arr[(m >= amount / 2) & (m < amount)] = 255
    return Image.fromarray(arr)


def d_jpeg(img, rng, s):
    quality = int(round(lerp(60, 10, s)))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def d_blur(img, rng, s):
    return img.filter(ImageFilter.GaussianBlur(radius=lerp(0.5, 2.5, s)))


def d_pixelate(img, rng, s):
    f = int(round(lerp(2, 6, s)))
    w, h = img.size
    small = img.resize((max(1, w // f), max(1, h // f)), Image.Resampling.BOX)
    return small.resize((w, h), Image.Resampling.NEAREST)


def d_rect(img, rng, s):
    w, h = img.size
    frac = lerp(0.10, 0.25, s)
    rw = max(4, int(w * frac * rng.uniform(0.7, 1.3)))
    rh = max(4, int(h * frac * rng.uniform(0.7, 1.3)))
    x = int(rng.integers(0, max(1, w - rw)))
    y = int(rng.integers(0, max(1, h - rh)))
    shade = 0 if rng.random() < 0.4 else int(rng.integers(90, 181))
    out = img.copy()
    ImageDraw.Draw(out).rectangle([x, y, x + rw, y + rh], fill=(shade, shade, shade))
    return out


def d_fade(img, rng, s):
    out = ImageEnhance.Contrast(img).enhance(lerp(0.8, 0.4, s))
    return Image.blend(out, Image.new("RGB", out.size, (235, 232, 222)), lerp(0.1, 0.4, s))


def d_grey(img, rng, s):
    grey = ImageOps.grayscale(img).convert("RGB")
    out = Image.blend(img, grey, lerp(0.7, 1.0, s))
    return ImageEnhance.Contrast(out).enhance(rng.uniform(0.6, 1.1))


def d_invert(img, rng, s):
    return ImageOps.invert(img)


def d_hicontrast(img, rng, s):
    out = ImageEnhance.Contrast(img).enhance(lerp(1.8, 4.0, s))
    return ImageEnhance.Brightness(out).enhance(rng.uniform(0.8, 1.3))


def d_streaks(img, rng, s):
    arr = np.asarray(img, dtype=np.float32)
    h, w, _ = arr.shape
    col = rng.normal(0.0, lerp(6, 28, s), (1, w, 1)).astype(np.float32)  # faint vertical lines
    shift = int(rng.integers(3, 12))
    alpha = lerp(0.15, 0.45, s)                                          # ghost copy shifted sideways
    out = (1 - alpha) * arr + alpha * np.roll(arr, shift, axis=1) + col
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def d_snow(img, rng, s):
    out = img.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    for _ in range(int(lerp(30, 250, s))):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        r, v = int(rng.integers(1, 4)), int(rng.integers(200, 256))
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(v, v, v))
    return out


def d_colorpatch(img, rng, s):
    out = img.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    for _ in range(int(lerp(2, 8, s))):
        pw, ph = int(rng.integers(6, 31)), int(rng.integers(6, 31))
        x, y = int(rng.integers(0, w - pw)), int(rng.integers(0, h - ph))
        color = tuple(int(c) for c in rng.integers(0, 256, 3))
        draw.rectangle([x, y, x + pw, y + ph], fill=color)
    return out


def d_sketch(img, rng, s):
    grey = ImageOps.grayscale(img)
    edges = ImageOps.invert(grey.filter(ImageFilter.FIND_EDGES))
    return Image.blend(grey, edges, lerp(0.5, 0.9, s)).convert("RGB")


DAMAGES = {
    "noise": d_noise, "saltpepper": d_saltpepper, "jpeg": d_jpeg, "blur": d_blur,
    "pixelate": d_pixelate, "rect": d_rect, "fade": d_fade,
    # --- new in this script ---
    "grey": d_grey, "invert": d_invert, "hicontrast": d_hicontrast, "streaks": d_streaks,
    "snow": d_snow, "colorpatch": d_colorpatch, "sketch": d_sketch,
}
ORIGINAL_DAMAGES = ["noise", "saltpepper", "jpeg", "blur", "pixelate", "rect", "fade"]
DAMAGE_NAMES = list(DAMAGES)


# =============================================================================
# DATASETS
# =============================================================================
class TrainSet(Dataset):
    """Training images with random flips/crops/colour changes plus 0-2 random damages."""

    def __init__(self, paths, labels, img_size):
        self.paths, self.labels = list(paths), list(labels)
        self.geo = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.ColorJitter(0.2, 0.2, 0.2),
        ])
        self.post = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
        self.rng = None  # created inside each worker process, seeded differently per worker

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        if self.rng is None:
            self.rng = np.random.default_rng(torch.initial_seed() % (2 ** 32))
        img = self.geo(Image.open(self.paths[i]).convert("RGB"))
        k = int(self.rng.choice([0, 1, 2], p=[0.15, 0.50, 0.35]))
        for name in self.rng.choice(DAMAGE_NAMES, size=k, replace=False):
            img = DAMAGES[str(name)](img, self.rng, float(self.rng.random()))
        return self.post(img), torch.tensor(self.labels[i], dtype=torch.float32)


class EvalSet(Dataset):
    """Validation images: clean, or with ONE fixed damage at fixed strength (same every epoch)."""

    def __init__(self, paths, labels, damage=None, severity=0.6):
        self.paths, self.labels = list(paths), list(labels)
        self.damage, self.severity = damage, severity
        self.post = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.damage:
            rng = np.random.default_rng(10_000 + i)
            img = DAMAGES[self.damage](img, rng, self.severity)
        return self.post(img), torch.tensor(self.labels[i], dtype=torch.float32)


# =============================================================================
# HELPERS
# =============================================================================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_auc(y, p):
    try:
        return float(roc_auc_score(y, p))
    except ValueError:  # only one class present (tiny --limit runs)
        return float("nan")


@torch.no_grad()
def predict(model, loader, device, use_amp):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x).squeeze(1)
        probs.append(torch.sigmoid(logits.float()).cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def build_model(pretrained):
    weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b3(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_features, 1))
    return model


def find_images(root):
    """Map image_id (filename without extension) -> full path for train/real and train/fake."""
    found = {}
    for sub in ("real", "fake"):
        for p in (Path(root) / "train" / sub).glob("*"):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                found[p.stem] = p
    return found


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", default="runs/b3_aug")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--smoothing", type=float, default=0.05, help="label smoothing (targets 0.05/0.95)")
    ap.add_argument("--patience", type=int, default=3, help="stop after this many epochs without improvement")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits", default="runs/b3/splits.csv",
                    help="reuse this split file if it exists, so every model is tested on the same photos")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--damaged_eval_n", type=int, default=0,
                    help="use only this many val images for the damaged sets (0 = all); lower = faster epochs")
    ap.add_argument("--limit", type=int, default=0, help="use only this many labelled images (quick tests)")
    ap.add_argument("--no_pretrained", action="store_true", help="skip ImageNet weights (debugging only)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if use_amp else "  <-- no GPU found!"))

    # ---- data: read labels, find files, split -------------------------------
    df = pd.read_csv(Path(args.data_root) / "train_metadata.csv")
    df["image_id"] = df["image_id"].astype(str).str.strip()
    files = find_images(args.data_root)
    missing = [i for i in df["image_id"] if i not in files]
    if missing:
        raise SystemExit(f"{len(missing)} images from train_metadata.csv not found, e.g. {missing[:3]}")
    if args.limit:
        df, _ = train_test_split(df, train_size=args.limit, stratify=df["label"], random_state=args.seed)

    split_path = out / (f"splits_limit{args.limit}.csv" if args.limit else "splits.csv")
    if not split_path.exists() and not args.limit and Path(args.splits).exists():
        shutil.copy(args.splits, split_path)  # same exam photos as the B0 run -> fair comparison
        print(f"copied split from {args.splits}")
    if split_path.exists():
        sp = pd.read_csv(split_path)
        print(f"reusing split from {split_path}")
    else:
        tr, va = train_test_split(df, test_size=args.val_frac, stratify=df["label"], random_state=args.seed)
        sp = pd.concat([tr.assign(split="train"), va.assign(split="val")]).reset_index(drop=True)
        sp.to_csv(split_path, index=False)
    tr_df, va_df = sp[sp.split == "train"], sp[sp.split == "val"]
    print(f"train: {len(tr_df)} images   val: {len(va_df)} images")

    tr_paths = [files[i] for i in tr_df["image_id"]]
    va_paths = [files[i] for i in va_df["image_id"]]
    tr_labels, va_labels = tr_df["label"].tolist(), va_df["label"].tolist()

    g = torch.Generator().manual_seed(args.seed)
    pin = use_amp
    persistent = args.num_workers > 0
    train_loader = DataLoader(TrainSet(tr_paths, tr_labels, args.img_size), batch_size=args.batch_size,
                              shuffle=True, drop_last=True, num_workers=args.num_workers,
                              pin_memory=pin, persistent_workers=persistent, generator=g)

    def make_eval_loader(paths, labels, damage=None):
        return DataLoader(EvalSet(paths, labels, damage), batch_size=64, shuffle=False,
                          num_workers=args.num_workers, pin_memory=pin, persistent_workers=persistent)

    val_loader = make_eval_loader(va_paths, va_labels)
    if args.damaged_eval_n and args.damaged_eval_n < len(va_paths):
        idx = np.random.RandomState(args.seed).choice(len(va_paths), args.damaged_eval_n, replace=False)
        d_paths, d_labels = [va_paths[i] for i in idx], [va_labels[i] for i in idx]
    else:
        d_paths, d_labels = va_paths, va_labels
    dmg_loader = DataLoader(ConcatDataset([EvalSet(d_paths, d_labels, name) for name in DAMAGE_NAMES]),
                            batch_size=64, shuffle=False, num_workers=args.num_workers,
                            pin_memory=pin, persistent_workers=persistent)

    # ---- model, loss, optimiser, schedule -----------------------------------
    model = build_model(pretrained=not args.no_pretrained).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    total_steps = args.epochs * len(train_loader)
    warm = max(1, len(train_loader))  # warm up for the first epoch

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        prog = (step - warm) / max(1, total_steps - warm)
        return max(0.01, 0.5 * (1 + math.cos(math.pi * prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    log_rows, start_epoch, best_score, best_epoch, bad = [], 0, -1.0, -1, 0
    if args.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best_score, best_epoch, bad = ck["epoch"] + 1, ck["best_score"], ck["best_epoch"], ck["bad"]
        log_rows = ck["log"]
        print(f"resumed from epoch {start_epoch}")

    # ---- training loop --------------------------------------------------------
    eps = args.smoothing
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()
        total_loss, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            target = y * (1 - 2 * eps) + eps  # label smoothing: 0 -> 0.05, 1 -> 0.95
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
            total_loss += loss.item() * len(y)
            n += len(y)

        p, yv = predict(model, val_loader, device, use_amp)
        auc_clean, acc_clean = safe_auc(yv, p), float(accuracy_score(yv, p >= 0.5))
        pd_all, yd_all = predict(model, dmg_loader, device, use_amp)
        n_d = len(d_paths)
        dmg = {name: safe_auc(yd_all[i * n_d:(i + 1) * n_d], pd_all[i * n_d:(i + 1) * n_d])
               for i, name in enumerate(DAMAGE_NAMES)}
        dmg_mean = float(np.nanmean(list(dmg.values())))  # all 14 kinds
        dmg_old = float(np.nanmean([dmg[k] for k in ORIGINAL_DAMAGES]))  # the 7 used in the B0/B3 runs
        score = 0.7 * auc_clean + 0.3 * dmg_mean  # mirrors the judges' 50 (accuracy) : 20 (robustness)
        secs = time.time() - t0

        row = {"epoch": epoch + 1, "train_loss": round(total_loss / max(1, n), 4),
               "val_auc_clean": round(auc_clean, 4), "val_acc_clean": round(acc_clean, 4)}
        row.update({f"auc_{k}": round(v, 4) for k, v in dmg.items()})
        row.update({"val_auc_damaged_mean": round(dmg_mean, 4), "auc_damaged_orig7_mean": round(dmg_old, 4),
                    "score": round(score, 4),
                    "lr": round(opt.param_groups[0]["lr"], 7), "epoch_time_s": round(secs, 1)})
        log_rows.append(row)
        pd.DataFrame(log_rows).to_csv(out / "training_log.csv", index=False)

        print(f"epoch {epoch + 1}/{args.epochs}  loss {row['train_loss']:.4f}  "
              f"AUC clean {auc_clean:.4f}  damaged(all 14) {dmg_mean:.4f}  damaged(orig 7) {dmg_old:.4f}  "
              f"score {score:.4f}  ({secs:.0f}s)")
        names = list(dmg)
        print("   old : " + "  ".join(f"{k} {dmg[k]:.3f}" for k in names[:7]))
        print("   new : " + "  ".join(f"{k} {dmg[k]:.3f}" for k in names[7:]))

        if score > best_score + 1e-4:
            best_score, best_epoch, bad = score, epoch + 1, 0
            torch.save(model.state_dict(), out / "best.pt")
            print("   -> new best, saved best.pt")
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best_score": best_score,
                    "best_epoch": best_epoch, "bad": bad, "log": log_rows}, out / "last.pt")
        if bad >= args.patience:
            print(f"no improvement for {args.patience} epochs - stopping early")
            break

    # ---- final report with the best model -------------------------------------
    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    p, yv = predict(model, val_loader, device, use_amp)
    pd.DataFrame({"image_id": va_df["image_id"].values, "label": (p >= 0.5).astype(int),
                  "confidence": p, "true_label": yv.astype(int)}).to_csv(out / "val_predictions.csv", index=False)
    auc, f1, acc = safe_auc(yv, p), float(f1_score(yv, p >= 0.5)), float(accuracy_score(yv, p >= 0.5))

    # speed: model only, batch of 64, random input
    model.eval()
    xb = torch.randn(64, 3, args.img_size, args.img_size, device=device)
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=use_amp):
        for _ in range(5):
            model(xb)
        if use_amp:
            torch.cuda.synchronize()
        t = time.time()
        for _ in range(20):
            model(xb)
        if use_amp:
            torch.cuda.synchronize()
    ms_per_img = (time.time() - t) / (20 * 64) * 1000

    summary = {
        "model": "efficientnet_b3 + 7 extra damages", "best_epoch": best_epoch, "best_score": round(best_score, 4),
        "val_auc_clean": round(auc, 4), "val_f1": round(f1, 4), "val_accuracy": round(acc, 4),
        "params_millions": round(sum(q.numel() for q in model.parameters()) / 1e6, 2),
        "weights_mb": round((out / "best.pt").stat().st_size / 1e6, 1),
        "inference_ms_per_image_batch64": round(ms_per_img, 3),
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 1e6) if use_amp else None,
        "seed": args.seed, "args": vars(args),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n===== FINAL (best epoch) =====")
    for k in ("best_epoch", "val_auc_clean", "val_f1", "val_accuracy", "params_millions",
              "weights_mb", "inference_ms_per_image_batch64", "peak_gpu_memory_mb"):
        print(f"{k}: {summary[k]}")
    print(f"\nfiles written to {out}/")


if __name__ == "__main__":
    main()
