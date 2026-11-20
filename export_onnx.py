"""
export_onnx.py - Export the complete 6-model ensemble to ONNX format.

Features:
- Bundles all 6 backbones (2x B3, 2x ConvNeXt, 2x Swin-T) into one graph.
- Includes Test-Time Augmentation (TTA horizontal flip averaging) directly inside the ONNX graph.
- Traces dynamic batch size on input ('image') and output ('prob_fake').
- Validates the exported graph with onnx.checker and verifies numerical accuracy with onnxruntime.

Usage:
    python export_onnx.py --weights ./model_weights.pt --output ./model.onnx
"""

import argparse
import time
import torch
import torch.nn as nn

from inference_script import load_members


class EnsembleONNX(nn.Module):
    """Unified 6-model ensemble forward pass with TTA horizontal-flip averaging."""
    def __init__(self, nets):
        super().__init__()
        self.nets = nn.ModuleList(nets)

    def forward(self, x):
        # x: (B, 3, 224, 224)
        x_flip = torch.flip(x, dims=[3])
        total = 0.0
        for net in self.nets:
            z = net(x).squeeze(1)
            z_flip = net(x_flip).squeeze(1)
            total = total + (z + z_flip) / 2.0
        mean_logit = total / len(self.nets)
        return torch.sigmoid(mean_logit)


def main():
    parser = argparse.ArgumentParser(description="Export 6-model deepfake ensemble to ONNX")
    parser.add_argument("--weights", default="model_weights.pt", help="Path to PyTorch model_weights.pt")
    parser.add_argument("--output", default="model.onnx", help="Path to output ONNX file")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    args = parser.parse_args()

    print(f"[1/4] Loading models from {args.weights}...")
    t0 = time.time()
    nets = load_members(args.weights, None, torch.device("cpu"))
    for net in nets:
        net.eval()

    model = EnsembleONNX(nets).eval()
    dummy = torch.randn(1, 3, 224, 224)

    print(f"[2/4] Testing PyTorch forward pass...")
    with torch.no_grad():
        pt_out = model(dummy)
    print(f"      PyTorch reference output: {pt_out.item():.6f}")

    print(f"[3/4] Exporting to {args.output} (opset {args.opset}, legacy TorchScript)...")
    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["image"],
        output_names=["prob_fake"],
        dynamic_axes={"image": {0: "batch"}, "prob_fake": {0: "batch"}},
        opset_version=args.opset,
        dynamo=False
    )
    print(f"      Successfully exported {args.output} in {time.time() - t0:.1f}s")

    print(f"[4/4] Verifying with ONNX Runtime...")
    try:
        import onnx
        import onnxruntime as ort

        onnx_model = onnx.load(args.output)
        onnx.checker.check_model(onnx_model)
        print("      ONNX graph integrity check: PASSED!")

        sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {"image": dummy.numpy()})[0]
        diff = abs(pt_out.item() - float(ort_out[0]))
        print(f"      ONNX Runtime output: {float(ort_out[0]):.6f} (Diff vs PyTorch: {diff:.8f})")
        assert diff < 1e-4, f"Tolerance exceeded: {diff}"
        print("      [SUCCESS] ONNX model verified bit-exact within single-precision tolerance!")
    except ImportError:
        print("      [NOTE] onnx or onnxruntime not installed in evaluation environment, skipping local check.")


if __name__ == "__main__":
    main()
