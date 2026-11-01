"""
package_models.py - bundle trained models into ONE weights file for inference_script.py.

Each --member is  <architecture>=<path to that run's best.pt>.  The architecture is the model name used
for training (efficientnet_b0, efficientnet_b3, convnext_tiny, swin_t, ...). Two runs of the same
architecture are fine (e.g. runs\\b3 and runs\\b3_aug are both efficientnet_b3).

Example (Windows, one line):
    python package_models.py --out model_weights.pt --member efficientnet_b0=runs\\b0_baseline\\best.pt --member efficientnet_b3=runs\\b3\\best.pt --member efficientnet_b3=runs\\b3_aug\\best.pt --member convnext_tiny=runs\\convnext_tiny\\best.pt --member swin_t=runs\\swin_t\\best.pt

Each member is loaded into its architecture first, so a wrong name or a wrong file fails here, not on the
judges' computer. Needs inference_script.py in the same folder.
"""
import argparse
from pathlib import Path

import torch

from inference_script import MODEL_NAMES, build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="model_weights.pt")
    ap.add_argument("--member", action="append", required=True, metavar="ARCH=PATH")
    args = ap.parse_args()

    members = []
    for spec in args.member:
        if "=" not in spec:
            raise SystemExit(f"--member must look like architecture=path, got '{spec}'")
        name, path = spec.split("=", 1)
        if name not in MODEL_NAMES:
            raise SystemExit(f"'{name}' is not a known architecture. Choose from {MODEL_NAMES}")
        state = torch.load(path, map_location="cpu")
        build_model(name).load_state_dict(state)  # raises if the file does not fit the architecture
        members.append({"name": name, "state_dict": state})
        print(f"ok  {name:18s} <- {path}  ({Path(path).stat().st_size / 1e6:.1f} MB)")

    torch.save({"format": "ensemble_v1", "models": members}, args.out)
    print(f"\nwrote {args.out}: {len(members)} model(s), {Path(args.out).stat().st_size / 1e6:.1f} MB")
    print("Test it:  python inference_script.py --test_dir <folder> --weights", args.out)


if __name__ == "__main__":
    main()
