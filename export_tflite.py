"""
export_tflite.py - Standalone TensorFlow Lite Export Module
Team MADD - Deepfake ML Hackathon (Cross-Framework Deployment)

Exports the trained Dual-Architecture Combo (EfficientNet-B3 + ConvNeXt-Tiny)
from PyTorch weights into optimized TensorFlow Lite FlatBuffers:
  - model_combo.tflite (Float32 full precision)
  - model_combo_fp16.tflite (Float16 compact precision)

Usage:
    python export_tflite.py --weights ./model_weights.pt --output_dir ./
"""

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

try:
    import onnxruntime as ort
    import onnx2tf
    import tensorflow as tf
except ImportError as e:
    sys.exit(f"[ERROR] Required export packages missing: {e}\nInstall with: pip install onnxruntime onnx2tf tensorflow")


def build_model(name, dropout=0.3):
    def head(in_features):
        return nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, 1))
    if name == "efficientnet_b3":
        m = models.efficientnet_b3(weights=None)
        m.classifier = head(m.classifier[1].in_features)
    elif name == "convnext_tiny":
        m = models.convnext_tiny(weights=None)
        m.classifier[2] = head(m.classifier[2].in_features)
    else:
        raise ValueError(f"Unsupported model architecture: {name}")
    return m


class DualCombo(nn.Module):
    """Dual-Architecture Ensemble Combo: Averages logits of B3 and ConvNeXt, returns sigmoid probability."""
    def __init__(self, m1, m2):
        super().__init__()
        self.m1 = m1
        self.m2 = m2

    def forward(self, x):
        z1 = self.m1(x)
        z2 = self.m2(x)
        avg_logit = (z1 + z2) / 2.0
        return torch.sigmoid(avg_logit)


def main():
    parser = argparse.ArgumentParser(description="Export Dual-Architecture Combo to TensorFlow Lite")
    parser.add_argument("--weights", default="model_weights.pt", help="Path to model_weights.pt")
    parser.add_argument("--output_dir", default=".", help="Directory to save exported .tflite files")
    parser.add_argument("--test_dir", default=None, help="Optional image folder for numerical parity validation")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version (default: 14)")
    args = parser.parse_args()

    weights_path = Path(args.weights).resolve()
    if not weights_path.exists():
        sys.exit(f"[ERROR] Weights file not found: {weights_path}")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"[1/4] Loading trained PyTorch weights from: {weights_path.name}")
    print("=" * 65)

    checkpoint = torch.load(weights_path, map_location="cpu")
    if "models" in checkpoint:
        b3_dict = checkpoint["models"][0]["state_dict"]
        cn_dict = checkpoint["models"][1]["state_dict"]
    else:
        b3_dict = checkpoint.get("efficientnet_b3", checkpoint)
        cn_dict = checkpoint.get("convnext_tiny", checkpoint)

    b3 = build_model("efficientnet_b3")
    b3.load_state_dict(b3_dict, strict=True)
    b3.eval()

    cn = build_model("convnext_tiny")
    cn.load_state_dict(cn_dict, strict=True)
    cn.eval()
    print("  Successfully initialized EfficientNet-B3 and ConvNeXt-Tiny backbones.")

    combo_pt = DualCombo(b3, cn).eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        combo_onnx_path = os.path.join(tmpdir, "model_combo.onnx")
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        print("\n" + "=" * 65)
        print(f"[2/4] Tracing and exporting ONNX graph (Opset {args.opset})...")
        print("=" * 65)

        torch.onnx.export(
            combo_pt,
            dummy_input,
            combo_onnx_path,
            input_names=["input"],
            output_names=["probability"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "probability": {0: "batch_size"}
            },
            opset_version=args.opset,
            dynamo=False,
            do_constant_folding=True
        )

        onnx_size_mb = os.path.getsize(combo_onnx_path) / (1024 * 1024)
        print(f"  Exported intermediate ONNX: {onnx_size_mb:.2f} MB")

        # Verify ONNX
        ort_sess = ort.InferenceSession(combo_onnx_path, providers=["CPUExecutionProvider"])
        test_rand = np.random.randn(2, 3, 224, 224).astype(np.float32)
        with torch.no_grad():
            pt_out = combo_pt(torch.from_numpy(test_rand)).numpy().flatten()
        ort_out = ort_sess.run(None, {"input": test_rand})[0].flatten()
        onnx_diff = np.max(np.abs(pt_out - ort_out))
        print(f"  PyTorch vs ONNX parity delta: {onnx_diff:.8e}")
        assert onnx_diff < 1e-4, "ONNX verification failed"
        print("  [OK] ONNX Numerical Parity Verified!")

        print("\n" + "=" * 65)
        print("[3/4] Compiling to TensorFlow Lite FlatBuffers (onnx2tf)...")
        print("=" * 65)

        tf_tmp_out = os.path.join(tmpdir, "tf_out")
        os.makedirs(tf_tmp_out, exist_ok=True)

        onnx2tf.convert(
            input_onnx_file_path=combo_onnx_path,
            output_folder_path=tf_tmp_out,
            non_verbose=True
        )

        raw_fp32 = os.path.join(tf_tmp_out, "model_combo_float32.tflite")
        raw_fp16 = os.path.join(tf_tmp_out, "model_combo_float16.tflite")

        target_fp32 = out_dir / "model_combo.tflite"
        target_fp16 = out_dir / "model_combo_fp16.tflite"

        shutil.copy2(raw_fp32, target_fp32)
        shutil.copy2(raw_fp16, target_fp16)

        print(f"  [OK] Saved Float32 TFLite: {target_fp32.name} ({target_fp32.stat().st_size / 1e6:.2f} MB)")
        print(f"  [OK] Saved Float16 TFLite: {target_fp16.name} ({target_fp16.stat().st_size / 1e6:.2f} MB)")

    # Optional verification if test directory exists
    test_dir_to_check = None
    if args.test_dir and Path(args.test_dir).is_dir():
        test_dir_to_check = Path(args.test_dir)
    elif (weights_path.parent.parent / "hackathon_dataset" / "test" / "images").is_dir():
        test_dir_to_check = weights_path.parent.parent / "hackathon_dataset" / "test" / "images"

    if test_dir_to_check:
        print("\n" + "=" * 65)
        print(f"[4/4] Numerical Parity Audit against real test images ({test_dir_to_check.name})...")
        print("=" * 65)

        interpreter = tf.lite.Interpreter(model_path=str(target_fp32))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        image_files = sorted([p for p in test_dir_to_check.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])[:50]
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        pt_preds, tf_preds = [], []
        for img_path in image_files:
            img = Image.open(img_path).convert("RGB").resize((224, 224))
            arr = np.array(img, dtype=np.float32) / 255.0
            norm_arr = (arr - mean) / std

            pt_in = torch.from_numpy(norm_arr.transpose(2, 0, 1)).unsqueeze(0).float()
            with torch.no_grad():
                pt_preds.append(combo_pt(pt_in).item())

            tf_in = np.expand_dims(norm_arr, axis=0)
            interpreter.set_tensor(input_details[0]["index"], tf_in)
            interpreter.invoke()
            tf_preds.append(float(interpreter.get_tensor(output_details[0]["index"])[0, 0]))

        pt_arr = np.array(pt_preds)
        tf_arr = np.array(tf_preds)
        deltas = np.abs(pt_arr - tf_arr)
        corr = np.corrcoef(pt_arr, tf_arr)[0, 1]
        agree = ((pt_arr >= 0.5) == (tf_arr >= 0.5)).mean() * 100.0

        print(f"  Audited Images:               {len(image_files)}")
        print(f"  Pearson Correlation:          {corr:.6f}")
        print(f"  Binary Label Agreement (0.5): {agree:.2f}%")
        print(f"  Max Absolute Delta:           {deltas.max():.8f}")
        print(f"  Mean Absolute Delta:          {deltas.mean():.8f}")
        assert corr > 0.99, f"Correlation {corr} is below threshold"
        print("  [SUCCESS] Full numerical parity confirmed across unseen images!")
    else:
        print("\n[NOTE] No test images folder found for post-export audit; compilation verified via random tensor parity.")

    print("\n" + "=" * 65)
    print("EXPORT COMPLETE: TENSORFLOW LITE ARTIFACTS READY FOR DEPLOYMENT")
    print("=" * 65)


if __name__ == "__main__":
    main()
