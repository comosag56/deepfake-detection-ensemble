"""
gradcam.py - Ultimate Hybrid Grad-CAM Interpretability Generator.
Deepfake ML Hackathon — IIIT Bangalore.

Combines multi-architecture ensemble comparison (EfficientNet-B3 vs. ConvNeXt-Tiny vs. Ensemble)
with deep error analysis across three distinct evaluation categories:
  1. Confident Correct: FAKE
  2. Confident Correct: REAL
  3. Confident MISTAKES / Ambiguous Boundary Cases

Self-contained: Requires ONLY torch, torchvision, pillow, numpy, pandas.
Zero external visualization packages needed (no matplotlib or cv2).

Usage:
    python gradcam.py
    python gradcam.py --weights model_weights.pt --out gradcam_visualizations.png
    python gradcam.py --test_dir ./test/images/ --out gradcam_visualizations.png
"""
import argparse
import sys
from pathlib import Path

# Ensure local module directory is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from inference_script import MEAN, STD, build_model

IMAGE_SIZE = 224


# =============================================================================
# COLORMAP & OVERLAY (Pure NumPy & PIL)
# =============================================================================
def jet_colormap(v):
    """Full-spectrum piecewise linear Jet colormap [0, 1] -> uint8 RGB [0, 255]."""
    v = np.clip(v, 0.0, 1.0)
    xp = [0.0, 0.125, 0.375, 0.625, 0.875, 1.0]
    fp_r = [0, 0, 0, 255, 255, 128]
    fp_g = [0, 0, 255, 255, 0, 0]
    fp_b = [143, 255, 255, 0, 0, 0]
    r = np.interp(v, xp, fp_r)
    g = np.interp(v, xp, fp_g)
    b = np.interp(v, xp, fp_b)
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def blend_heatmap(orig_pil, cam_norm, alpha=0.52):
    """Overlays Jet heatmap onto original 224x224 PIL Image."""
    orig_sq = orig_pil.resize((IMAGE_SIZE, IMAGE_SIZE)).convert("RGB")
    heat_rgb = jet_colormap(cam_norm)
    heat_pil = Image.fromarray(heat_rgb)
    return Image.blend(orig_sq, heat_pil, alpha=alpha)


# =============================================================================
# GRAD-CAM ENGINE
# =============================================================================
class GradCAM:
    """Computes Grad-CAM for a specified target convolutional layer."""

    def __init__(self, model, target_layer, device, layout="nchw"):
        self.model = model.to(device).eval()
        self.target_layer = target_layer
        self.device = device
        self.layout = layout
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def __call__(self, x_tensor):
        self.model.zero_grad()
        out = self.model(x_tensor.to(self.device))
        z = out.squeeze()
        prob = torch.sigmoid(z).item()

        # Evidence for the predicted class
        target_score = z if prob >= 0.5 else -z
        target_score.backward(retain_graph=False)

        act = self.activations
        grad = self.gradients
        if self.layout == "nhwc":
            act = act.permute(0, 3, 1, 2)
            grad = grad.permute(0, 3, 1, 2)

        # Global average pool the gradients
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * act).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()

        denom = cam.max() - cam.min()
        if denom > 1e-8:
            cam = (cam - cam.min()) / denom
        else:
            cam = np.zeros_like(cam)

        return prob, cam


def load_model_from_weights(weights_path, target_arch):
    """Loads matching architecture weights from single checkpoint or ensemble bundle."""
    obj = torch.load(weights_path, map_location="cpu")
    if isinstance(obj, dict) and "models" in obj:
        for m in reversed(obj["models"]):
            if m["name"] == target_arch:
                net = build_model(target_arch)
                net.load_state_dict(m["state_dict"])
                return net
        raise SystemExit(f"No architecture '{target_arch}' found in {weights_path}")
    elif isinstance(obj, dict) and "name" in obj and obj["name"] == target_arch:
        net = build_model(target_arch)
        net.load_state_dict(obj["state_dict"])
        return net
    elif isinstance(obj, dict):
        net = build_model(target_arch)
        net.load_state_dict(obj)
        return net
    raise SystemExit(f"Unrecognized format in {weights_path}")


# =============================================================================
# DRAWING UTILITIES
# =============================================================================
def draw_badge(draw, text, xy, font, bg_color=(20, 20, 20), fg_color=(255, 255, 255)):
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 3
    rect = [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad]
    draw.rectangle(rect, fill=bg_color)
    draw.text((x, y), text, fill=fg_color, font=font)


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="Deepfake ML Hackathon - Ultimate Grad-CAM Generator")
    ap.add_argument("--weights", default="model_weights.pt", help="path to model_weights.pt")
    ap.add_argument("--images_dir", default=None, help="folder with images (val or test)")
    ap.add_argument("--test_dir", default=None, help="alias for --images_dir")
    ap.add_argument("--labels_csv", default=None, help="optional labels CSV (image_id, label)")
    ap.add_argument("--n_per_group", type=int, default=3, help="number of images per group")
    ap.add_argument("--out", default="gradcam_visualizations.png", help="output comparison sheet")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Initializing Grad-CAM engine on: {device}")

    # Resolve weights path
    weights_path = Path(args.weights)
    if not weights_path.exists():
        candidates = [
            Path("model_weights.pt"),
            Path(__file__).parent / "model_weights.pt",
            Path("daksh_trial_v2/model_weights.pt"),
            Path("submission_v3/model_weights.pt"),
        ]
        for c in candidates:
            if c.exists():
                weights_path = c
                break
    if not weights_path.exists():
        raise SystemExit(f"Weights file '{args.weights}' not found.")

    # 1. Load two diverse complementary backbones
    print(f"[INFO] Extracting EfficientNet-B3 and ConvNeXt-Tiny from {weights_path}...")
    model_b3 = load_model_from_weights(weights_path, "efficientnet_b3")
    model_cnx = load_model_from_weights(weights_path, "convnext_tiny")

    cam_b3 = GradCAM(model_b3, model_b3.features[-1], device, layout="nchw")
    cam_cnx = GradCAM(model_cnx, model_cnx.features[-1], device, layout="nchw")

    # Resolve images folder
    target_dir = args.images_dir or args.test_dir
    img_dir = None
    if target_dir and Path(target_dir).is_dir():
        img_dir = Path(target_dir)
    else:
        for cand in [
            Path("test/images"),
            Path("test"),
            Path("val_images"),
            Path("data/test"),
            Path("hackathon_dataset/test/images"),
            Path("scratch/test_5_images"),
            Path("."),
        ]:
            if cand.is_dir() and any(p.suffix.lower() in {".jpg", ".jpeg", ".png"} for p in cand.iterdir()):
                img_dir = cand
                break
    if img_dir is None:
        raise SystemExit("No folder containing images found. Please provide --test_dir <path_to_images>")
    print(f"[INFO] Using images from: {img_dir}")

    # Check for labels
    labels_file = Path(args.labels_csv) if args.labels_csv else Path("val_labels.csv")
    has_labels = labels_file.exists()
    df_lab = None
    if has_labels:
        try:
            df_lab = pd.read_csv(labels_file)[["image_id", "label"]]
            df_lab["image_id"] = df_lab["image_id"].astype(str).str.strip()
            print(f"[INFO] Loaded ground truth labels from: {labels_file}")
        except Exception:
            has_labels = False

    tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    # Find valid images
    valid_records = []
    if has_labels and df_lab is not None:
        for _, r in df_lab.iterrows():
            stem = r['image_id']
            for ext in ('.jpg', '.jpeg', '.png'):
                p_img = img_dir / f"{stem}{ext}"
                if p_img.exists():
                    valid_records.append((stem, int(r['label']), p_img))
                    break
    if not valid_records:
        has_labels = False
        all_imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])[:60]
        for p in all_imgs:
            valid_records.append((p.stem, -1, p))

    if not valid_records:
        raise SystemExit(f"No valid images found in '{img_dir}'")

    print(f"[INFO] Evaluating {len(valid_records)} photos (ground-truth available: {has_labels})...")
    results = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(valid_records), batch_size):
            chunk = valid_records[i:i + batch_size]
            tensors = torch.stack([tf(Image.open(p).convert("RGB")) for _, _, p in chunk]).to(device)
            p_b3_batch = torch.sigmoid(model_b3(tensors).squeeze(1)).cpu().numpy()
            p_cnx_batch = torch.sigmoid(model_cnx(tensors).squeeze(1)).cpu().numpy()

            for j, (img_id, label, p_img) in enumerate(chunk):
                p_b3 = float(p_b3_batch[j])
                p_cnx = float(p_cnx_batch[j])
                z_b3 = np.log(np.clip(p_b3, 1e-6, 1 - 1e-6) / (1 - np.clip(p_b3, 1e-6, 1 - 1e-6)))
                z_cnx = np.log(np.clip(p_cnx, 1e-6, 1 - 1e-6) / (1 - np.clip(p_cnx, 1e-6, 1 - 1e-6)))
                p_ens = float(1.0 / (1.0 + np.exp(-(z_b3 + z_cnx) / 2.0)))
                pred_lbl = 1 if p_ens >= 0.5 else 0
                results.append({
                    "image_id": img_id,
                    "path": p_img,
                    "label": label,
                    "p_b3": p_b3,
                    "p_cnx": p_cnx,
                    "p_ens": p_ens,
                    "pred": pred_lbl,
                    "is_correct": (pred_lbl == label) if has_labels else True,
                    "confidence_err": abs(p_ens - label) if has_labels else abs(p_ens - 0.5),
                })

    df_res = pd.DataFrame(results)
    k = min(args.n_per_group, max(1, len(df_res) // 3))

    if has_labels:
        top_fakes = df_res[(df_res.is_correct) & (df_res.label == 1)].sort_values("p_ens", ascending=False).head(k)
        top_reals = df_res[(df_res.is_correct) & (df_res.label == 0)].sort_values("p_ens", ascending=True).head(k)
        top_mistakes = df_res[~df_res.is_correct].sort_values("confidence_err", ascending=False).head(k)
        groups = [
            ("CONFIDENT CORRECT: FAKE (LABEL = 1)", top_fakes, (220, 60, 60), "Highest predicted fake probability on confirmed deepfakes"),
            ("CONFIDENT CORRECT: REAL (LABEL = 0)", top_reals, (40, 160, 80), "Lowest predicted fake probability on confirmed authentic faces"),
            ("CONFIDENT MISTAKES & AMBIGUOUS BOUNDARY CASES", top_mistakes, (230, 140, 30), "Largest discrepancy between prediction and true label")
        ]
    else:
        top_fakes = df_res.sort_values("p_ens", ascending=False).head(k)
        top_reals = df_res.sort_values("p_ens", ascending=True).head(k)
        top_mistakes = df_res.iloc[(df_res["p_ens"] - 0.5).abs().argsort()].head(k)
        groups = [
            ("CONFIDENT PREDICTED: FAKE (P >= 0.5)", top_fakes, (220, 60, 60), "Highest predicted fake probability by ensemble"),
            ("CONFIDENT PREDICTED: REAL (P < 0.5)", top_reals, (40, 160, 80), "Lowest predicted fake probability by ensemble"),
            ("HIGH UNCERTAINTY: BOUNDARY CASES (P ~ 0.5)", top_mistakes, (230, 140, 30), "Predictions closest to decision boundary")
        ]

    # Canvas Construction
    header_main_h = 56
    section_banner_h = 32
    col_header_h = 28
    col_w = IMAGE_SIZE + 8
    row_h = IMAGE_SIZE + 8
    pad = 14
    total_w = pad * 2 + col_w * 4

    total_rows = sum(len(g[1]) for g in groups)
    total_h = header_main_h + (len(groups) * (section_banner_h + col_header_h)) + (total_rows * row_h) + pad * 2

    canvas = Image.new("RGB", (total_w, total_h), (14, 16, 20))
    draw = ImageDraw.Draw(canvas)

    try:
        f_title = ImageFont.truetype("arial.ttf", 16)
        f_sec = ImageFont.truetype("arial.ttf", 13)
        f_col = ImageFont.truetype("arial.ttf", 12)
        f_tag = ImageFont.truetype("arial.ttf", 11)
    except IOError:
        f_title = f_sec = f_col = f_tag = ImageFont.load_default()

    draw.text((pad, 12), "DEEPFAKE FORENSIC INTERPRETABILITY: MULTI-ARCHITECTURE GRAD-CAM", fill=(255, 215, 0), font=f_title)
    draw.text((pad, 34), "Grad-CAM heat-maps of the predicted class, validated models", fill=(160, 175, 195), font=f_col)

    current_y = header_main_h

    for g_title, g_df, banner_color, g_desc in groups:
        if len(g_df) == 0:
            continue
        draw.rectangle([pad, current_y + 4, total_w - pad, current_y + section_banner_h - 2], fill=(24, 28, 36))
        draw.rectangle([pad, current_y + 4, pad + 6, current_y + section_banner_h - 2], fill=banner_color)
        draw.text((pad + 14, current_y + 8), g_title, fill=banner_color, font=f_sec)
        draw.text((pad + 480, current_y + 10), f"({g_desc})", fill=(150, 160, 175), font=f_tag)
        current_y += section_banner_h

        col_titles = ["Original", "EfficientNet-B3 (CNN)", "ConvNeXt-Tiny (Modern CNN)", "Average of B3 & ConvNeXt"]
        for c_i, title in enumerate(col_titles):
            draw.text((pad + c_i * col_w + 2, current_y + 6), title, fill=(185, 205, 225), font=f_col)
        current_y += col_header_h

        for _, row in g_df.iterrows():
            raw_pil = Image.open(row["path"]).convert("RGB")
            x_ten = tf(raw_pil).unsqueeze(0)

            p_b3, cam_b3_arr = cam_b3(x_ten)
            p_cnx, cam_cnx_arr = cam_cnx(x_ten)
            cam_ens_arr = (cam_b3_arr + cam_cnx_arr) / 2.0
            p_ens = row["p_ens"]

            orig_sq = raw_pil.resize((IMAGE_SIZE, IMAGE_SIZE))
            ov_b3 = blend_heatmap(orig_sq, cam_b3_arr)
            ov_cnx = blend_heatmap(orig_sq, cam_cnx_arr)
            ov_ens = blend_heatmap(orig_sq, cam_ens_arr)

            if has_labels and row["label"] in (0, 1):
                true_txt = "TRUE: FAKE" if row["label"] == 1 else "TRUE: REAL"
                true_bg = (180, 30, 30) if row["label"] == 1 else (25, 130, 45)
            else:
                true_txt = "PRED: FAKE" if p_ens >= 0.5 else "PRED: REAL"
                true_bg = (180, 30, 30) if p_ens >= 0.5 else (25, 130, 45)

            canvas.paste(orig_sq, (pad + 0 * col_w, current_y))
            draw_badge(draw, true_txt, (pad + 0 * col_w + 4, current_y + 4), f_tag, bg_color=true_bg)
            draw_badge(draw, str(row["image_id"])[:10], (pad + 0 * col_w + 4, current_y + IMAGE_SIZE - 16), f_tag, bg_color=(0, 0, 0))

            canvas.paste(ov_b3, (pad + 1 * col_w, current_y))
            draw_badge(draw, f"B3 P(Fake): {p_b3:.2f}", (pad + 1 * col_w + 4, current_y + 4), f_tag)

            canvas.paste(ov_cnx, (pad + 2 * col_w, current_y))
            draw_badge(draw, f"ConvNeXt P(Fake): {p_cnx:.2f}", (pad + 2 * col_w + 4, current_y + 4), f_tag)

            canvas.paste(ov_ens, (pad + 3 * col_w, current_y))
            verdict = "PRED: FAKE" if p_ens >= 0.5 else "PRED: REAL"
            v_bg = (160, 20, 20) if p_ens >= 0.5 else (20, 120, 30)
            draw_badge(draw, f"{verdict} ({p_ens * 100:.1f}%)", (pad + 3 * col_w + 4, current_y + 4), f_tag, bg_color=v_bg)

            current_y += row_h

        current_y += 6

    canvas.save(args.out)
    print(f"\n[DONE] Successfully saved Ultimate Hybrid Grad-CAM to: {args.out}")
    print(f"  Dimensions: {total_w}x{total_h} pixels")


if __name__ == "__main__":
    main()
