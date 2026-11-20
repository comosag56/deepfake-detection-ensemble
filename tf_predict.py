"""
tf_predict.py - Standalone TensorFlow Lite Inference Module
Team MADD - Deepfake ML Hackathon (Cross-Framework Deployment)

Evaluates images using the exported dual-architecture TensorFlow Lite FlatBuffer (model_combo.tflite).
Requires ZERO PyTorch dependencies (runs solely on numpy, pillow, and tensorflow/tflite_runtime).

Usage:
    python tf_predict.py --test_dir ./test/images/ --model ./model_combo.tflite --output tf_predictions.csv
"""

import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

try:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            sys.exit("[ERROR] Neither tensorflow, tflite_runtime, nor ai_edge_litert is installed.")

IMAGE_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
VALID_EXTS = {".jpg", ".jpeg", ".png"}


def preprocess_image(path):
    img = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    norm = (arr - MEAN) / STD
    return np.expand_dims(norm, axis=0)  # [1, 224, 224, 3] NHWC


def main():
    parser = argparse.ArgumentParser(description="TensorFlow Lite Standalone Inference")
    parser.add_argument("--test_dir", required=True, help="Folder containing test images")
    parser.add_argument("--model", default="model_combo.tflite", help="Path to .tflite model")
    parser.add_argument("--output", default="tf_predictions.csv", help="Path for output predictions CSV")
    parser.add_argument("--threshold", type=float, default=0.50, help="Classification decision threshold")
    parser.add_argument("--tta_flip", action="store_true", help="Enable horizontal flip test-time augmentation")
    args = parser.parse_args()

    test_path = Path(args.test_dir)
    if not test_path.is_dir():
        sys.exit(f"[ERROR] Test directory '{args.test_dir}' does not exist.")

    image_paths = sorted([p for p in test_path.iterdir() if p.suffix.lower() in VALID_EXTS])
    if not image_paths:
        sys.exit(f"[ERROR] No valid images found in '{args.test_dir}'.")

    if not os.path.exists(args.model):
        sys.exit(f"[ERROR] Model file '{args.model}' not found.")

    print(f"[INFO] Loading TensorFlow Lite model: {args.model}")
    interpreter = Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    in_idx = input_details[0]["index"]
    out_idx = output_details[0]["index"]

    print(f"[INFO] Total images to process: {len(image_paths)}")
    print(f"[INFO] Model input shape: {input_details[0]['shape']}, TTA flip: {'ON' if args.tta_flip else 'OFF'}")

    ids, confidences = [], []
    t0 = time.time()

    for idx, p in enumerate(image_paths, 1):
        x = preprocess_image(p)
        interpreter.set_tensor(in_idx, x)
        interpreter.invoke()
        prob = float(interpreter.get_tensor(out_idx)[0, 0])

        if args.tta_flip:
            x_flip = np.flip(x, axis=2)  # flip width dimension
            interpreter.set_tensor(in_idx, x_flip)
            interpreter.invoke()
            prob_flip = float(interpreter.get_tensor(out_idx)[0, 0])
            prob = (prob + prob_flip) / 2.0

        prob = float(np.clip(prob, 0.0, 1.0))
        ids.append(p.stem)
        confidences.append(prob)

        if idx % 500 == 0 or idx == len(image_paths):
            elapsed = time.time() - t0
            ms_per_img = (elapsed / idx) * 1000
            print(f"  Processed {idx}/{len(image_paths)} images ({ms_per_img:.1f} ms/img)...")

    labels = (np.array(confidences) >= args.threshold).astype(int)
    df = pd.DataFrame({
        "image_id": ids,
        "label": labels,
        "confidence": confidences
    })

    assert len(df) == len(image_paths), "Row count mismatch"
    assert df["image_id"].is_unique, "Duplicate image IDs"
    assert df["confidence"].between(0.0, 1.0).all(), "Confidence out of bounds"

    df.to_csv(args.output, index=False)
    elapsed = time.time() - t0
    print(f"[DONE] Saved predictions to {args.output}")
    print(f"  Total time: {elapsed:.2f}s ({elapsed / len(image_paths) * 1000:.1f} ms/image)")
    print(f"  Predicted Fake: {labels.sum()} ({labels.mean()*100:.1f}%) | Real: {(labels==0).sum()}")
    print(f"  Mean Confidence: {np.mean(confidences):.4f}")


if __name__ == "__main__":
    main()
