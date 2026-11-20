"""
onnx_predict.py - Standalone ONNX Runtime Inference Script (Zero PyTorch Dependency)

This script demonstrates cross-framework deployment readiness by running deepfake
inference entirely on ONNX Runtime and NumPy, without requiring PyTorch.

Usage:
    python onnx_predict.py --test_dir ./test/images --model ./model.onnx --output predictions_onnx.csv
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

try:
    import onnxruntime as ort
except ImportError:
    raise SystemExit("[ERROR] onnxruntime is required. Install with: pip install onnxruntime")

IMAGE_SIZE = (224, 224)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
EXTS = {".jpg", ".jpeg", ".png"}


def preprocess_image(path: Path) -> np.ndarray:
    """Loads, resizes to 224x224, and normalizes image with ImageNet mean/std."""
    img = Image.open(path).convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (224, 224, 3)
    arr = np.transpose(arr, (2, 0, 1))               # (3, 224, 224)
    return arr


def main():
    parser = argparse.ArgumentParser(description="Standalone ONNX Runtime Deepfake Inference")
    parser.add_argument("--test_dir", required=True, help="Folder containing test facial images")
    parser.add_argument("--model", default="model.onnx", help="Path to exported model.onnx")
    parser.add_argument("--output", default="predictions_onnx.csv", help="Path to save predictions CSV")
    parser.add_argument("--batch_size", type=int, default=32, help="Inference batch size")
    parser.add_argument("--threshold", type=float, default=0.50, help="Decision threshold for label 1")
    args = parser.parse_args()

    test_folder = Path(args.test_dir)
    if not test_folder.is_dir():
        raise SystemExit(f"[ERROR] Directory does not exist: '{args.test_dir}'")

    image_paths = sorted([p for p in test_folder.iterdir() if p.suffix.lower() in EXTS])
    if not image_paths:
        raise SystemExit(f"[ERROR] No valid images found in '{args.test_dir}'")

    print(f"[INFO] Initializing ONNX Runtime session: '{args.model}'...")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(args.model, providers=providers)
    print(f"[INFO] Active execution provider: {sess.get_providers()[0]}")

    ids = []
    confs = []
    t0 = time.time()

    for i in range(0, len(image_paths), args.batch_size):
        batch_paths = image_paths[i:i + args.batch_size]
        batch_tensors = np.stack([preprocess_image(p) for p in batch_paths], axis=0)  # (B, 3, 224, 224)
        batch_norm = (batch_tensors - MEAN) / STD

        probs = sess.run(None, {"image": batch_norm})[0]  # Shape: (B,)
        confs.extend(probs.tolist())
        ids.extend([p.stem for p in batch_paths])

        if (i // args.batch_size) % 10 == 0 or (i + args.batch_size) >= len(image_paths):
            print(f"  Processed {len(ids)} / {len(image_paths)} images...")

    total_time = time.time() - t0
    confs = np.array(confs, dtype=np.float32)
    labels = (confs >= args.threshold).astype(int)

    out_df = pd.DataFrame({
        "image_id": ids,
        "label": labels,
        "confidence": confs
    })

    out_df.to_csv(args.output, index=False)
    print(f"\n[SUCCESS] Wrote predictions to '{args.output}'")
    print(f"  Total images processed: {len(out_df)}")
    print(f"  Predicted FAKE (1): {out_df['label'].sum()} ({out_df['label'].mean() * 100:.2f}%)")
    print(f"  Predicted REAL (0): {(out_df['label'] == 0).sum()} ({(out_df['label'] == 0).mean() * 100:.2f}%)")
    print(f"  Mean Confidence: {out_df['confidence'].mean():.4f}")
    print(f"  Total Time: {total_time:.2f}s ({total_time / len(out_df) * 1000:.2f} ms/image)")


if __name__ == "__main__":
    main()
