#!/usr/bin/env python3
"""Measure DRISHTI on a held-out scene, against baselines.

    python scripts/evaluate.py --ckpt checkpoints/drishti/best.pt \
        --scene data/raw/surat_s2l2a.tif --out outputs/eval

Everything reported here is measured on a scene the model never trained on,
under Wald's protocol: the scene is degraded 4x, each method reconstructs it,
and the reconstruction is compared against the original -- which is real
Sentinel-2 data, not a synthetic target.

Baselines are not optional. A super-resolution number without a bicubic column
next to it says nothing, because bicubic already scores respectably and a model
that merely blurs can beat a bad one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drishti.data import RGBN, degrade, load_scene, split_scene, upsample  # noqa: E402
from drishti.metrics import evaluate_all  # noqa: E402
from drishti.model.drishti import Drishti  # noqa: E402
from drishti.tiling import Stitcher, iter_tiles  # noqa: E402


def tiled_sr(model, lr: torch.Tensor, scale: int, tile: int = 128, overlap: int = 16,
             device=None, passes: int = 0):
    """Run the model over a whole scene in tiles, feather-blending the result.

    passes > 0 additionally returns the MC-dropout standard deviation, which is
    the confidence raster.
    """
    _, c, h, w = lr.shape
    st = Stitcher(bands=c, sr_height=h * scale, sr_width=w * scale, scale=scale, ramp=overlap)
    st_u = Stitcher(bands=1, sr_height=h * scale, sr_width=w * scale, scale=scale, ramp=overlap) if passes else None
    st_g = Stitcher(bands=1, sr_height=h * scale, sr_width=w * scale, scale=scale, ramp=overlap)
    tiles = iter_tiles(h, w, tile, overlap)
    for i, t in enumerate(tiles):
        patch = lr[:, :, t.y0 : t.y0 + t.h, t.x0 : t.x0 + t.w].to(device)
        with torch.no_grad():
            if passes:
                sr, unc = model.predict_with_uncertainty(patch, passes=passes)
                st_u.add(t, unc[0].cpu().numpy())
            else:
                sr = model(patch)
            _, gate = model(patch, return_gate=True)
        # the gate lives on the LR grid; put it on the SR grid for display
        st_g.add(t, upsample(gate, scale)[0].cpu().numpy())
        st.add(t, sr[0].cpu().numpy())
        if (i + 1) % 25 == 0 or i + 1 == len(tiles):
            print(f"  tile {i+1}/{len(tiles)}", flush=True)
    out = torch.from_numpy(st.result()).unsqueeze(0)
    unc = torch.from_numpy(st_u.result()).unsqueeze(0) if passes else None
    gate = torch.from_numpy(st_g.result()).unsqueeze(0)
    return out, unc, gate


def run_opensr(lr, sr, hr, bic, scale: int, crop: int = 256) -> dict:
    """opensr-test metrics on a centre crop, DRISHTI and bicubic side by side.

    Band order is RGBN (B02 B03 B04 B08); opensr-test's spectral term assumes a
    particular ordering, so treat `spectral` as indicative. The hallucination /
    omission / improvement fractions are the headline numbers here.
    """
    import warnings

    import opensr_test

    warnings.filterwarnings("ignore")
    _, _, h, w = hr.shape
    c = min(crop, h, w)
    c -= c % scale
    y, x = (h - c) // 2, (w - c) // 2
    yl, xl, cl = y // scale, x // scale, c // scale
    hr_c = hr[0, :, y:y + c, x:x + c]
    lr_c = lr[0, :, yl:yl + cl, xl:xl + cl]

    def one(pred):
        m = opensr_test.Metrics()
        d = m.compute(lr=lr_c, sr=pred[0, :, y:y + c, x:x + c], hr=hr_c)
        return {k: round(float(v), 5) for k, v in d.items()}

    return {"crop_px": c, "drishti": one(sr), "bicubic": one(bic)}


def save_png(arr: np.ndarray, path: Path, pct: tuple[float, float] = (2, 98)) -> None:
    """RGB quicklook with a percentile stretch. Display only -- every number in
    the report comes from the reflectance arrays, never from these PNGs."""
    from PIL import Image

    rgb = arr[[2, 1, 0]]  # B04, B03, B02
    lo, hi = np.percentile(rgb, pct[0]), np.percentile(rgb, pct[1])
    out = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)
    Image.fromarray((out.transpose(1, 2, 0) * 255).astype("uint8")).save(path)


def save_gray(arr: np.ndarray, path: Path, cmap_amber: bool = False) -> None:
    from PIL import Image

    a = arr[0]
    lo, hi = np.percentile(a, 2), np.percentile(a, 98)
    n = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    if cmap_amber:
        img = np.stack([n, n * 0.69, n * 0.13], axis=-1)
    else:
        img = np.stack([n, n, n], axis=-1)
    Image.fromarray((img * 255).astype("uint8")).save(path)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--scene", required=True, help="held-out scene, never trained on")
    p.add_argument("--out", default="outputs/eval")
    p.add_argument("--tile", type=int, default=128)
    p.add_argument("--overlap", type=int, default=16)
    p.add_argument("--passes", type=int, default=8, help="MC-dropout passes for confidence")
    p.add_argument("--device", default="auto")
    p.add_argument("--max-size", type=int, default=1024,
                   help="crop the scene to at most this many pixels a side")
    p.add_argument("--spatial-holdout", type=float, default=None, metavar="FRAC",
                   help="evaluate only the southern FRAC of the scene -- use the same "
                        "value the model was trained with, so this is the region it "
                        "never saw")
    p.add_argument("--gap", type=int, default=64)
    p.add_argument("--no-opensr", action="store_true",
                   help="skip opensr-test hallucination/omission/improvement metrics")
    args = p.parse_args()

    if args.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available()
                           else "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        dev = torch.device(args.device)

    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    a = ck.get("args", {})
    scale = a.get("scale", 4)
    sigma = a.get("sigma", None)
    model = Drishti(bands=len(RGBN), ch=a.get("ch", 64), cnn_blocks=a.get("cnn_blocks", 8),
                    swin_depth=a.get("swin_depth", 4), window=a.get("window", 8),
                    scale=scale, sigma=sigma, dropout=a.get("dropout", 0.1)).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    sc = load_scene(args.scene, RGBN)
    if args.spatial_holdout is not None:
        _, sc = split_scene(sc, args.spatial_holdout, args.gap)
        print(f"evaluating held-out region only: southern {args.spatial_holdout:.0%}, "
              f"{args.gap}-row gap")
    arr = sc.array
    if args.max_size and (arr.shape[1] > args.max_size or arr.shape[2] > args.max_size):
        y = (arr.shape[1] - args.max_size) // 2
        x = (arr.shape[2] - args.max_size) // 2
        arr = arr[:, y : y + args.max_size, x : x + args.max_size]
    H, W = arr.shape[1] - arr.shape[1] % scale, arr.shape[2] - arr.shape[2] % scale
    hr = torch.from_numpy(np.ascontiguousarray(arr[:, :H, :W])).unsqueeze(0)
    lr = degrade(hr, scale, sigma)

    print(f"scene   : {sc.name}  HR {tuple(hr.shape[-2:])}  LR {tuple(lr.shape[-2:])}")
    print(f"model   : {args.ckpt}  (step {ck.get('step')})  device {dev}")
    print("running DRISHTI...", flush=True)
    sr, unc, gate = tiled_sr(model, lr, scale, args.tile, args.overlap, dev, args.passes)

    results = {
        "bicubic": evaluate_all(upsample(lr, scale), hr, lr, scale, sigma),
        "drishti": evaluate_all(sr, hr, lr, scale, sigma),
    }

    # opensr-test: hallucination / omission / improvement on a centre crop. These
    # are the "did the model invent detail that isn't real" numbers. Guarded --
    # a failure here must not sink the core report.
    opensr = None
    if not args.no_opensr:
        try:
            opensr = run_opensr(lr, sr, hr, upsample(lr, scale), scale)
        except Exception as exc:  # noqa: BLE001
            print(f"(opensr-test skipped: {exc})")




    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    save_png(hr[0].numpy(), out / f"{sc.name}_truth_10m.png")
    save_png(upsample(lr, scale)[0].numpy(), out / f"{sc.name}_bicubic.png")
    save_png(sr[0].numpy(), out / f"{sc.name}_drishti.png")
    save_png(lr[0].numpy(), out / f"{sc.name}_input_40m.png")
    save_gray(gate[0].numpy(), out / f"{sc.name}_gate.png")
    if unc is not None:
        save_gray(unc[0].numpy(), out / f"{sc.name}_confidence.png", cmap_amber=True)

    deg_desc = ("area mean (box filter)" if not sigma
                else f"Gaussian sigma={sigma} + area mean")
    report = {
        "protocol": f"Wald: scene degraded {scale}x ({deg_desc}), reconstructed, "
                    "compared against the original Sentinel-2 pixels",
        "scene": sc.name,
        "scene_path": str(args.scene),
        "held_out": True,
        "split": ("spatial: southern %.0f%%, %d-row gap" % (args.spatial_holdout*100, args.gap))
                 if args.spatial_holdout is not None else "separate region",
        "checkpoint": str(args.ckpt),
        "train_step": ck.get("step"),
        "bands": RGBN,
        "hr_shape": list(hr.shape[-2:]),
        "scale": scale,
        "mc_dropout_passes": args.passes,
        "results": results,
        "opensr_test": opensr,
        "gain_over_bicubic_db": results["drishti"]["psnr_db"] - results["bicubic"]["psnr_db"],
    }
    (out / f"{sc.name}_report.json").write_text(json.dumps(report, indent=2))

    rows = [("metric", "unit", "bicubic", "DRISHTI")]
    for key, unit in [("psnr_db", "dB"), ("ssim", "-"), ("sam_deg", "deg"),
                      ("rmse_reflectance", "refl"), ("ndvi_mae", "-"),
                      ("consistency_blockavg_rmse", "refl"), ("consistency_rmse", "refl")]:
        rows.append((key, unit, f"{results['bicubic'][key]:.4g}", f"{results['drishti'][key]:.4g}"))
    md = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * 4]
    md += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    md.append("")
    if opensr:
        md.append(f"opensr-test (centre {opensr['crop_px']} px crop):")
        md.append("")
        md.append("| metric | bicubic | DRISHTI |")
        md.append("|---|---|---|")
        for k in ("im_metric", "om_metric", "ha_metric", "reflectance", "spectral"):
            if k in opensr["drishti"]:
                md.append(f"| {k} | {opensr['bicubic'][k]:.4g} | {opensr['drishti'][k]:.4g} |")
        md.append("")
        md.append("`im_metric` = improvement, `om_metric` = omission, "
                  "`ha_metric` = hallucination (all fractions of high-gradient pixels).")
        md.append("")
    md.append(f"Held-out scene: **{sc.name}** — never used in training. "
              f"PSNR gain over bicubic: **{report['gain_over_bicubic_db']:+.3f} dB**.")
    (out / f"{sc.name}_report.md").write_text("\n".join(md))

    print("\n" + "\n".join(md))
    print(f"\nwritten to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
