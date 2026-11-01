"""
gradcam.py - Ultimate Hybrid Grad-CAM Interpretability Generator.
Deepfake ML Hackathon — IIIT Bangalore.

Combines multi-architecture ensemble comparison (EfficientNet-B3 vs. ConvNeXt-Tiny vs. Ensemble)
with deep error analysis across three distinct evaluation categories:
  1. Confident Correct: FAKE
  2. Confident Correct: REAL
  3. Confident MISTAKES

Self-contained: Requires ONLY torch, torchvision, pillow, numpy, pandas.
Zero external visualization packages needed (no matplotlib or cv2).

Usage:
    python gradcam.py
    python gradcam.py --weights submission_v3/model_weights.pt --out gradcam_showcase.png
    python gradcam.py --n_per_group 2 --out gradcam_compact.png
"""
import argparse
from pathlib import Path

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

        # Evidence for the predicted class (positive gradient for predicted outcome)
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
    """Loads matching architecture weights from single checkpoint or ensemble bundle.
    Traverses in reverse to select the validated models (80% split) rather than full-dataset retrained models."""
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
    ap.add_argument("--weights", default="submission_v3/model_weights.pt",
                    help="path to model_weights.pt")
    ap.add_argument("--images_dir", default="val_images", help="folder with validation images")
    ap.add_argument("--labels_csv", default="val_labels.csv", help="labels CSV (image_id, label)")
    ap.add_argument("--n_per_group", type=int, default=3,
                    help="number of images per group (Fakes, Reals, Mistakes)")
    ap.add_argument("--out", default="gradcam_visualizations.png", help="output comparison sheet")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Initializing Grad-CAM engine on: {device}")

    weights_path = Path(args.weights)
    if not weights_path.exists():
        if Path("model_weights.pt").exists():
            weights_path = Path("model_weights.pt")
        else:
            raise SystemExit(f"Weights file '{args.weights}' not found.")

    # 1. Load two diverse complementary backbones
    print(f"[INFO] Extracting EfficientNet-B3 and ConvNeXt-Tiny from {weights_path}...")
    model_b3 = load_model_from_weights(weights_path, "efficientnet_b3")
    model_cnx = load_model_from_weights(weights_path, "convnext_tiny")

    cam_b3 = GradCAM(model_b3, model_b3.features[-1], device, layout="nchw")
    cam_cnx = GradCAM(model_cnx, model_cnx.features[-1], device, layout="nchw")

    # 2. Score validation images to select top samples & edge cases
    img_dir = Path(args.images_dir)
    labels_file = Path(args.labels_csv)
    if not img_dir.exists() or not labels_file.exists():
        raise SystemExit(f"Images '{args.images_dir}' or labels '{args.labels_csv}' missing.")

    df_lab = pd.read_csv(labels_file)[["image_id", "label"]]
    df_lab["image_id"] = df_lab["image_id"].astype(str).str.strip()

    tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    print("[INFO] Evaluating validation photos in batches to isolate top fakes, reals, and mistakes...")
    results = []
    
    # Collect available image paths (defensive check for jpg, jpeg, png)
    valid_records = []
    for _, r in df_lab.iterrows():
        stem = r['image_id']
        for ext in ('.jpg', '.jpeg', '.png'):
            p_img = img_dir / f"{stem}{ext}"
            if p_img.exists():
                valid_records.append((stem, int(r['label']), p_img))
                break

    # Fast batched scoring (batch_size=64)
    batch_size = 64
    with torch.no_grad():
        for i in range(0, len(valid_records), batch_size):
            chunk = valid_records[i:i + batch_size]
            tensors = torch.stack([tf(Image.open(p).convert("RGB")) for _, _, p in chunk]).to(device)
            p_b3_batch = torch.sigmoid(model_b3(tensors).squeeze(1)).cpu().numpy()
            p_cnx_batch = torch.sigmoid(model_cnx(tensors).squeeze(1)).cpu().numpy()

            for j, (img_id, label, p_img) in enumerate(chunk):
                p_b3 = float(p_b3_batch[j])
                p_cnx = float(p_cnx_batch[j])
                # Logit blend for consensus
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
                    "is_correct": (pred_lbl == label),
                    "confidence_err": abs(p_ens - label),
                })

    df_res = pd.DataFrame(results)
    k = args.n_per_group

    # Category 1: Top Confident Correct Fakes (highest p_ens where label == 1)
    top_fakes = df_res[(df_res.is_correct) & (df_res.label == 1)].sort_values("p_ens", ascending=False).head(k)
    # Category 2: Top Confident Correct Reals (lowest p_ens where label == 0)
    top_reals = df_res[(df_res.is_correct) & (df_res.label == 0)].sort_values("p_ens", ascending=True).head(k)
    # Category 3: Top Confident Mistakes (highest confidence_err where pred != label)
    top_mistakes = df_res[~df_res.is_correct].sort_values("confidence_err", ascending=False).head(k)

    groups = [
        ("CONFIDENT CORRECT: FAKE (LABEL = 1)",
         top_fakes, (220, 60, 60), "Highest predicted fake probability on confirmed deepfakes"),
        ("CONFIDENT CORRECT: REAL (LABEL = 0)",
         top_reals, (40, 160, 80), "Lowest predicted fake probability on confirmed authentic faces"),
        ("CONFIDENT MISTAKES & AMBIGUOUS BOUNDARY CASES",
         top_mistakes, (230, 140, 30), "Largest discrepancy between prediction and true label")
    ]

    # 3. Canvas Construction
    header_main_h = 56
    section_banner_h = 32
    col_header_h = 28
    col_w = IMAGE_SIZE + 8
    row_h = IMAGE_SIZE + 8
    pad = 14
    total_w = pad * 2 + col_w * 4

    total_rows = len(top_fakes) + len(top_reals) + len(top_mistakes)
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

    # Main Canvas Header
    draw.text((pad, 12), "DEEPFAKE FORENSIC INTERPRETABILITY: MULTI-ARCHITECTURE GRAD-CAM",
              fill=(255, 215, 0), font=f_title)
    draw.text((pad, 34), "Grad-CAM heat-maps of the predicted class, validated models",
              fill=(160, 175, 195), font=f_col)

    current_y = header_main_h

    # 4. Render Groups
    for g_title, g_df, banner_color, g_desc in groups:
        # Section Banner
        draw.rectangle([pad, current_y + 4, total_w - pad, current_y + section_banner_h - 2], fill=(24, 28, 36))
        draw.rectangle([pad, current_y + 4, pad + 6, current_y + section_banner_h - 2], fill=banner_color)
        draw.text((pad + 14, current_y + 8), g_title, fill=banner_color, font=f_sec)
        draw.text((pad + 480, current_y + 10), f"({g_desc})", fill=(150, 160, 175), font=f_tag)
        current_y += section_banner_h

        # Column Subheaders
        col_titles = [
            "Original",
            "EfficientNet-B3 (CNN)",
            "ConvNeXt-Tiny (Modern CNN)",
            "Average of B3 & ConvNeXt"
        ]
        for c_i, title in enumerate(col_titles):
            draw.text((pad + c_i * col_w + 2, current_y + 6), title, fill=(185, 205, 225), font=f_col)
        current_y += col_header_h

        # Render Rows
        for _, row in g_df.iterrows():
            raw_pil = Image.open(row["path"]).convert("RGB")
            x_ten = tf(raw_pil).unsqueeze(0)

            # Grad-CAM calculations
            p_b3, cam_b3_arr = cam_b3(x_ten)
            p_cnx, cam_cnx_arr = cam_cnx(x_ten)
            cam_ens_arr = (cam_b3_arr + cam_cnx_arr) / 2.0
            p_ens = row["p_ens"]

            orig_sq = raw_pil.resize((IMAGE_SIZE, IMAGE_SIZE))
            ov_b3 = blend_heatmap(orig_sq, cam_b3_arr)
            ov_cnx = blend_heatmap(orig_sq, cam_cnx_arr)
            ov_ens = blend_heatmap(orig_sq, cam_ens_arr)

            true_txt = "TRUE: FAKE" if row["label"] == 1 else "TRUE: REAL"
            true_bg = (180, 30, 30) if row["label"] == 1 else (25, 130, 45)

            # Col 1: Original
            canvas.paste(orig_sq, (pad + 0 * col_w, current_y))
            draw_badge(draw, true_txt, (pad + 0 * col_w + 4, current_y + 4), f_tag, bg_color=true_bg)
            draw_badge(draw, str(row["image_id"])[:10], (pad + 0 * col_w + 4, current_y + IMAGE_SIZE - 16), f_tag,
                       bg_color=(0, 0, 0))

            # Col 2: EfficientNet-B3
            canvas.paste(ov_b3, (pad + 1 * col_w, current_y))
            draw_badge(draw, f"B3 P(Fake): {p_b3:.2f}", (pad + 1 * col_w + 4, current_y + 4), f_tag)

            # Col 3: ConvNeXt-Tiny
            canvas.paste(ov_cnx, (pad + 2 * col_w, current_y))
            draw_badge(draw, f"ConvNeXt P(Fake): {p_cnx:.2f}", (pad + 2 * col_w + 4, current_y + 4), f_tag)

            # Col 4: Average of B3 & ConvNeXt
            canvas.paste(ov_ens, (pad + 3 * col_w, current_y))
            verdict = "PRED: FAKE" if p_ens >= 0.5 else "PRED: REAL"
            v_bg = (160, 20, 20) if p_ens >= 0.5 else (20, 120, 30)
            draw_badge(draw, f"{verdict} ({p_ens * 100:.1f}%)", (pad + 3 * col_w + 4, current_y + 4), f_tag,
                       bg_color=v_bg)

            current_y += row_h

        current_y += 6

    canvas.save(args.out)
    print(f"\n[DONE] Successfully saved Ultimate Hybrid Grad-CAM to: {args.out}")
    print(f"  Dimensions: {total_w}x{total_h} pixels")
    print(f"  Visualized: {len(top_fakes)} Fakes, {len(top_reals)} Reals, {len(top_mistakes)} Confident Mistakes")
    print(f"  Columns: Original | EfficientNet-B3 | ConvNeXt-Tiny | Average of B3 & ConvNeXt")


if __name__ == "__main__":
    main()
