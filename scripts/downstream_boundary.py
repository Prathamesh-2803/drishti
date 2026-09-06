#!/usr/bin/env python3
"""Downstream task: does super-resolution recover field boundaries?

CLAUDE.md: "The headline number is the downstream task", not PSNR. Here the task
is field / parcel boundary delineation. Under Wald's protocol on a held-out
scene:

    truth   = the real 10 m Sentinel-2 scene
    LR      = truth degraded 4x (box filter)
    bicubic = LR upsampled 4x
    DRISHTI = LR super-resolved 4x by the model

We extract boundaries the same way from each reconstruction and from the truth,
then measure how well each reconstruction's boundaries agree with the truth's
(boundary F1 at a fixed pixel tolerance -- the standard BF score). If DRISHTI's
boundaries match the truth better than bicubic's do, the super-resolution carried
real, task-relevant information.

    python scripts/downstream_boundary.py --ckpt checkpoints/drishti/best.pt \
        --scene data/raw/vasai_s2l2a.tif --out outputs/eval_vasai

The boundary extractor (NDVI -> Canny) is deliberately simple and fixed across
all three inputs so the comparison is fair; it is not tuned per method.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drishti.data import RGBN, degrade, load_scene, upsample  # noqa: E402
from drishti.model.drishti import Drishti  # noqa: E402
from drishti.tiling import Stitcher, iter_tiles  # noqa: E402


def sr_tiled(model, lr: torch.Tensor, scale: int, device, tile=128, overlap=16) -> torch.Tensor:
    _, c, h, w = lr.shape
    st = Stitcher(c, h * scale, w * scale, scale, ramp=overlap)
    for t in iter_tiles(h, w, tile, overlap):
        patch = lr[:, :, t.y0:t.y0 + t.h, t.x0:t.x0 + t.w].to(device)
        with torch.no_grad():
            out = model(patch)
        st.add(t, out[0].cpu().numpy())
    return torch.from_numpy(st.result()).unsqueeze(0)


def ndvi_np(cube: np.ndarray) -> np.ndarray:
    """NDVI from an (RGBN, H, W) reflectance cube (red=idx 2, nir=idx 3)."""
    r, n = cube[2], cube[3]
    return (n - r) / (n + r + 1e-6)


def boundary_map(cube: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Fixed NDVI -> Canny edge map. Same thresholds for every input."""
    from skimage.feature import canny

    v = ndvi_np(cube)
    v = np.clip((v + 1.0) / 2.0, 0, 1)  # NDVI [-1,1] -> [0,1]
    return canny(v, sigma=1.0, low_threshold=lo, high_threshold=hi)


def bf_score(pred_edges: np.ndarray, true_edges: np.ndarray, tol: int = 2) -> dict:
    """Boundary F1: precision = fraction of predicted edge pixels within `tol` of
    a true edge; recall = fraction of true edge pixels within `tol` of a
    predicted edge."""
    from scipy.ndimage import distance_transform_edt

    if not true_edges.any() or not pred_edges.any():
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    d_true = distance_transform_edt(~true_edges)
    d_pred = distance_transform_edt(~pred_edges)
    precision = float((d_true[pred_edges] <= tol).mean())
    recall = float((d_pred[true_edges] <= tol).mean())
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--scene", required=True, help="held-out scene, never trained on")
    p.add_argument("--out", default="outputs/eval")
    p.add_argument("--max-size", type=int, default=1024)
    p.add_argument("--tol", type=int, default=2, help="boundary match tolerance (px)")
    p.add_argument("--canny-lo", type=float, default=0.10)
    p.add_argument("--canny-hi", type=float, default=0.20)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    dev = (torch.device(args.device) if args.device != "auto" else
           torch.device("cuda" if torch.cuda.is_available()
                        else "mps" if torch.backends.mps.is_available() else "cpu"))

    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    a = ck.get("args", {})
    scale, sigma = a.get("scale", 4), a.get("sigma", 0.0)
    model = Drishti(bands=len(RGBN), ch=a.get("ch", 64), cnn_blocks=a.get("cnn_blocks", 8),
                    swin_depth=a.get("swin_depth", 4), window=a.get("window", 8),
                    scale=scale, sigma=sigma, dropout=a.get("dropout", 0.1)).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    sc = load_scene(args.scene, RGBN)
    arr = sc.array
    if args.max_size and (arr.shape[1] > args.max_size or arr.shape[2] > args.max_size):
        y = (arr.shape[1] - args.max_size) // 2
        x = (arr.shape[2] - args.max_size) // 2
        arr = arr[:, y:y + args.max_size, x:x + args.max_size]
    H, W = arr.shape[1] - arr.shape[1] % scale, arr.shape[2] - arr.shape[2] % scale
    hr = torch.from_numpy(np.ascontiguousarray(arr[:, :H, :W])).unsqueeze(0)
    lr = degrade(hr, scale, sigma)

    bic = upsample(lr, scale)
    sr = sr_tiled(model, lr, scale, dev)

    lo, hi = args.canny_lo, args.canny_hi
    e_true = boundary_map(hr[0].numpy(), lo, hi)
    e_bic = boundary_map(bic[0].numpy(), lo, hi)
    e_sr = boundary_map(sr[0].numpy(), lo, hi)

    res = {
        "task": "field-boundary delineation (NDVI -> Canny), Wald protocol, held-out scene",
        "scene": sc.name,
        "checkpoint": str(args.ckpt),
        "train_step": ck.get("step"),
        "hr_shape": [H, W],
        "tolerance_px": args.tol,
        "canny": {"sigma": 1.0, "low": lo, "high": hi},
        "true_edge_fraction": round(float(e_true.mean()), 5),
        "bicubic_vs_truth": bf_score(e_bic, e_true, args.tol),
        "drishti_vs_truth": bf_score(e_sr, e_true, args.tol),
    }
    res["f1_gain_over_bicubic"] = round(
        res["drishti_vs_truth"]["f1"] - res["bicubic_vs_truth"]["f1"], 4)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"{sc.name}_boundary.json").write_text(json.dumps(res, indent=2))

    try:
        from PIL import Image
        for name, e in [("truth", e_true), ("bicubic", e_bic), ("drishti", e_sr)]:
            Image.fromarray((~e * 255).astype("uint8")).save(out / f"{sc.name}_edges_{name}.png")
    except Exception as exc:  # noqa: BLE001
        print(f"(edge PNGs skipped: {exc})")

    print(json.dumps(res, indent=2))
    print(f"\nboundary F1  bicubic {res['bicubic_vs_truth']['f1']:.4f}  ->  "
          f"DRISHTI {res['drishti_vs_truth']['f1']:.4f}   "
          f"({res['f1_gain_over_bicubic']:+.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
