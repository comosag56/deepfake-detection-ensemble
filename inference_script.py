"""
inference_script.py - Deepfake ML Hackathon submission script.

The judges run exactly this:
    python inference_script.py --test_dir ./test/images/ --weights ./model_weights.pt --output predictions.csv

--weights is ONE file. It is normally made by package_models.py and holds several trained models
(an ensemble). For each photo every model gives a score and we average them (default: average of the
models' log-odds, then converted to a probability of FAKE; --blend prob averages probabilities instead).
A plain state_dict from a single training run also works if you add --model <name>.

Output CSV columns: image_id (file name without extension), label (0 real / 1 fake), confidence (probability of FAKE).

Self-contained on purpose: needs only torch, torchvision, pillow, numpy, pandas (scikit-learn only for the
optional --labels_csv check). No internet is needed - no pretrained weights are downloaded.
No hardcoded paths. Works on GPU or CPU.

Optional:
    --labels_csv val_labels.csv   print AUC / accuracy too (used to check this script on your own validation photos)
    --tta_flip                    also average over horizontally flipped photos (slower, tiny gain)
    --fp16                        half precision on GPU (faster; default is full precision for exact reproducibility)
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

IMAGE_SIZE = 224
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
EXTS = {".jpg", ".jpeg", ".png"}
MODEL_NAMES = ["efficientnet_b0", "efficientnet_b3", "efficientnet_v2_s", "resnet50",
               "convnext_tiny", "swin_t", "vit_b_16"]


def build_model(name, dropout=0.3):
    """Same architectures as train_model.py. weights=None: the trained weights are loaded afterwards."""
    def head(in_features):
        return nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, 1))

    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier = head(m.classifier[1].in_features)
    elif name == "efficientnet_b3":
        m = models.efficientnet_b3(weights=None)
        m.classifier = head(m.classifier[1].in_features)
    elif name == "efficientnet_v2_s":
        m = models.efficientnet_v2_s(weights=None)
        m.classifier = head(m.classifier[1].in_features)
    elif name == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = head(m.fc.in_features)
    elif name == "convnext_tiny":
        m = models.convnext_tiny(weights=None)
        m.classifier[2] = head(m.classifier[2].in_features)
    elif name == "swin_t":
        m = models.swin_t(weights=None)
        m.head = head(m.head.in_features)
    elif name == "vit_b_16":
        m = models.vit_b_16(weights=None)
        m.heads.head = head(m.heads.head.in_features)
    else:
        raise SystemExit(f"Unknown model name '{name}'. Known: {MODEL_NAMES}")
    return m


class TestDataset(Dataset):
    """Every image in one flat folder (jpg / jpeg / png, any capitalisation)."""

    def __init__(self, image_dir):
        d = Path(image_dir)
        if not d.is_dir():
            raise SystemExit(f"--test_dir '{image_dir}' is not a folder")
        self.paths = sorted(p for p in d.iterdir() if p.suffix.lower() in EXTS)
        if not self.paths:
            raise SystemExit(f"No images found in '{image_dir}'")
        self.tf = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # no change for 224x224 photos
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img), self.paths[i].stem


def load_members(weights_path, fallback_name, device):
    obj = torch.load(weights_path, map_location="cpu")
    if isinstance(obj, dict) and "models" in obj:
        members = obj["models"]
    elif isinstance(obj, dict) and "name" in obj and "state_dict" in obj:
        members = [obj]
    elif isinstance(obj, dict):
        if not fallback_name:
            raise SystemExit("This weights file is a plain state_dict: add --model <name> (e.g. --model swin_t)")
        members = [{"name": fallback_name, "state_dict": obj}]
    else:
        raise SystemExit("Unrecognised weights file format")
    nets = []
    for mem in members:
        net = build_model(mem["name"])
        net.load_state_dict(mem["state_dict"])  # strict: fails loudly if the architecture does not match
        nets.append(net.to(device).eval())
    print(f"[INFO] loaded {len(nets)} model(s): {[m['name'] for m in members]} on {device}")
    return nets


@torch.no_grad()
def predict(nets, loader, device, fp16, tta, blend="logit"):
    """blend='logit': average each model's raw score (log-odds), then convert to a probability.
    blend='prob': average the probabilities directly."""
    ids, probs = [], []
    use_amp = fp16 and device.type == "cuda"
    for x, batch_ids in loader:
        x = x.to(device, non_blocking=True)
        total = 0
        for net in nets:
            with torch.autocast(device_type=device.type, enabled=use_amp):
                z = net(x).float().squeeze(1)
                if tta:
                    z_flip = net(torch.flip(x, dims=[3])).float().squeeze(1)
            if blend == "logit":
                score = (z + z_flip) / 2 if tta else z
            else:
                score = (torch.sigmoid(z) + torch.sigmoid(z_flip)) / 2 if tta else torch.sigmoid(z)
            total = total + score
        mean = total / len(nets)
        probs.append((torch.sigmoid(mean) if blend == "logit" else mean).cpu().numpy())
        ids.extend(list(batch_ids))
    return ids, np.concatenate(probs)


def main():
    ap = argparse.ArgumentParser(description="Deepfake ML Hackathon - inference script")
    ap.add_argument("--test_dir", required=True, help="folder with the test images")
    ap.add_argument("--weights", required=True, help="model weights file (one file)")
    ap.add_argument("--output", default="predictions.csv")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--model", default=None, help="only needed if --weights is a plain single-model state_dict")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--no_tta", action="store_true", help="disable horizontal flip test-time augmentation")
    ap.add_argument("--tta_flip", action="store_true", default=True, help="enable horizontal flip test-time augmentation (default: True)")
    ap.add_argument("--blend", choices=["logit", "prob"], default="logit",
                    help="how to combine models: average log-odds (default) or average probabilities")
    ap.add_argument("--labels_csv", default=None, help="optional csv with image_id,label to print AUC/accuracy")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    nets = load_members(args.weights, args.model, device)
    ds = TestDataset(args.test_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"))

    use_tta = not args.no_tta
    if not use_tta:
        print("[INFO] Test-Time Augmentation (TTA) disabled via --no_tta.")
    print(f"[INFO] {len(ds)} images, batch size {args.batch_size}, "
          f"{'fp16' if args.fp16 and device.type == 'cuda' else 'fp32'}, flip-averaging {'on' if use_tta else 'off'}, "
          f"blend: {args.blend}")

    t1 = time.time()
    ids, probs = predict(nets, loader, device, args.fp16, use_tta, args.blend)
    probs = np.clip(probs, 0.0, 1.0)
    t2 = time.time()

    out = pd.DataFrame({"image_id": ids, "label": (probs >= args.threshold).astype(int), "confidence": probs})
    assert len(out) == len(ds), "row count differs from image count"
    assert out["image_id"].is_unique, "duplicate image ids (two files with the same name but different extension?)"
    assert np.isfinite(out["confidence"]).all(), "NaN or infinite confidence"
    assert out["confidence"].between(0, 1).all() and out["label"].isin([0, 1]).all()
    out.to_csv(args.output, index=False)

    print(f"[DONE] wrote {args.output}")
    print(f"  images: {len(out)}   predicted FAKE: {out['label'].sum()} ({out['label'].mean() * 100:.1f}%)   "
          f"REAL: {(out['label'] == 0).sum()}   mean confidence: {out['confidence'].mean():.4f}")
    print(f"  time: loading models {t1 - t0:.1f}s, inference {t2 - t1:.1f}s ({(t2 - t1) / len(out) * 1000:.2f} ms/image)")

    if args.labels_csv:
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        lab = pd.read_csv(args.labels_csv)[["image_id", "label"]].rename(columns={"label": "true"})
        m = out.merge(lab, on="image_id")
        print(f"  check against {args.labels_csv}: {len(m)} labelled images")
        print(f"  AUC-ROC {roc_auc_score(m['true'], m['confidence']):.4f}   "
              f"accuracy {accuracy_score(m['true'], m['label']):.4f}   F1 {f1_score(m['true'], m['label']):.4f}")


if __name__ == "__main__":
    main()
