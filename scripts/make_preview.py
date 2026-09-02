#!/usr/bin/env python3
"""Export matched PNG previews for the swipe viewer from LR + SR COGs.

The LR image is upsampled x4 with nearest-neighbour so both PNGs share an
identical pixel grid and extent -- the swipe compares like for like. A single
2-98 percentile stretch is computed on the SR image and applied to BOTH
images so brightness differences are real, not artefacts of scaling.

Outputs into <viewer>/assets/:
    lr_rgb.png  sr_rgb.png            true colour  (B04 B03 B02)
    lr_nir.png  sr_nir.png            false colour (B08 B04 B03)
    uncertainty.png                   TTA spread, magma, own scale
    qa_mask.png                       red where bicubic fallback was used
    preview.json                      extent, scale, stretch values, QA summary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from matplotlib import cm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drishti.infer import BANDS  # noqa: E402

BAND_IDX = {b: i for i, b in enumerate(BANDS)}


def _stretch_params(cube: np.ndarray, bands: list[str]) -> dict[str, list[float]]:
    out = {}
    for b in bands:
        v = cube[BAND_IDX[b]]
        lo, hi = np.percentile(v, (2, 98))
        out[b] = [float(lo), float(hi)]
    return out


def _compose(cube: np.ndarray, bands: list[str], stretch: dict) -> np.ndarray:
    chans = []
    for b in bands:
        lo, hi = stretch[b]
        c = np.clip((cube[BAND_IDX[b]] - lo) / max(hi - lo, 1e-6), 0, 1)
        chans.append(c)
    return (np.stack(chans, axis=-1) * 255).astype(np.uint8)


def _save_png(arr: np.ndarray, path: Path) -> None:
    from PIL import Image

    Image.fromarray(arr).save(path)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lr", required=True, help="LR COG (10 m).")
    p.add_argument("--sr", required=True, help="SR COG (2.5 m).")
    p.add_argument(
        "--viewer",
        default="viewer",
        help="Viewer directory; assets go in <viewer>/assets/.",
    )
    args = p.parse_args()

    assets = Path(args.viewer) / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    with rasterio.open(args.sr) as sr_ds:
        sr = sr_ds.read().astype(np.float32)
        sf = float(sr_ds.tags().get("reflectance_scale_factor", 1.0) or 1.0)
        sr *= sf  # DN -> reflectance
        sr_bounds = sr_ds.bounds
        sr_crs = str(sr_ds.crs)
        sr_res = sr_ds.res
    with rasterio.open(args.lr) as lr_ds:
        lr = lr_ds.read().astype(np.float32)
        lr *= float(lr_ds.tags().get("reflectance_scale_factor", 1.0) or 1.0)

    scale = round(sr.shape[1] / lr.shape[1])
    lr_up = np.repeat(np.repeat(lr, scale, axis=1), scale, axis=2)
    # match shapes exactly (SR grid is the reference)
    lr_up = lr_up[:, : sr.shape[1], : sr.shape[2]]

    rgb = ["B04", "B03", "B02"]
    nir = ["B08", "B04", "B03"]
    stretch_rgb = _stretch_params(sr, rgb)
    stretch_nir = _stretch_params(sr, nir)

    _save_png(_compose(sr, rgb, stretch_rgb), assets / "sr_rgb.png")
    _save_png(_compose(lr_up, rgb, stretch_rgb), assets / "lr_rgb.png")
    _save_png(_compose(sr, nir, stretch_nir), assets / "sr_nir.png")
    _save_png(_compose(lr_up, nir, stretch_nir), assets / "lr_nir.png")

    summary: dict = {
        "sr_crs": sr_crs,
        "sr_bounds": [sr_bounds.left, sr_bounds.bottom, sr_bounds.right, sr_bounds.top],
        "sr_res_m": [abs(sr_res[0]), abs(sr_res[1])],
        "width": int(sr.shape[2]),
        "height": int(sr.shape[1]),
        "scale": scale,
        "stretch_rgb": stretch_rgb,
        "stretch_nir": stretch_nir,
    }

    unc_path = Path(args.sr).with_suffix("").as_posix() + "_uncertainty.tif"
    if Path(unc_path).exists():
        with rasterio.open(unc_path) as u_ds:
            u = u_ds.read(1).astype(np.float32) / 10000.0  # DN -> reflectance
            u[u == 0] = np.nan  # 0 = no estimate (QA fallback tile)
        finite = u[np.isfinite(u)]
        umax = float(np.percentile(finite, 98)) if finite.size else 1.0
        un = np.clip(np.nan_to_num(u, nan=umax) / max(umax, 1e-6), 0, 1)
        rgba = (cm.magma(un) * 255).astype(np.uint8)
        _save_png(rgba, assets / "uncertainty.png")
        summary["uncertainty_scale_max"] = umax

    qa_path = Path(args.sr).with_suffix("").as_posix() + "_qa_mask.tif"
    if Path(qa_path).exists():
        with rasterio.open(qa_path) as q_ds:
            q = q_ds.read(1)
        rgba = np.zeros((*q.shape, 4), dtype=np.uint8)
        rgba[q == 1] = [220, 30, 30, 160]
        _save_png(rgba, assets / "qa_mask.png")
        summary["qa_failed_fraction"] = float((q == 1).mean())

    qa_json = Path(f"{args.sr}.qa.json")
    if qa_json.exists():
        r = json.loads(qa_json.read_text())
        summary["qa"] = {
            k: r[k]
            for k in (
                "model",
                "n_tiles",
                "n_failed",
                "consistency_overall",
                "qa_thresholds",
                "uncertainty_method",
            )
            if k in r
        }

    (assets / "preview.json").write_text(json.dumps(summary, indent=2))
    print("wrote:", *(p.name for p in sorted(assets.glob("*"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
