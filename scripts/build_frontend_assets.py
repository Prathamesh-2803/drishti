#!/usr/bin/env python3
"""Render web assets + data.js for the DRISHTI explorer (frontend/).

Reads the per-scene products written by run_drishti.py (outputs/sr/<name>.tif and
its _confidence / _gate siblings) plus the held-out eval report, and writes:

    frontend/assets/<name>_{lr,sr,nir,conf,gate}.jpg    (13 scenes)
    frontend/data.js                                    window.DRISHTI = {...}

Run after scripts/phase_d.sh step 3 (or standalone once outputs/sr/ is populated).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "frontend" / "assets"
SRDIR = ROOT / "outputs/sr"
RAWDIR = ROOT / "data/raw"
EVAL = ROOT / "outputs/eval_vasai"

# name -> (short label terrain, one-line note)
TERRAIN = {
    "vasai": ("coastal wetlands & salt pans", "The held-out region — the model was trained without ever seeing it."),
    "mumbai": ("dense island metropolis", "High-rise cores, reclaimed land, the hardest test for a super-resolver."),
    "pune": ("Deccan plateau city", "Riverine suburbs against dry basalt uplands."),
    "surat": ("Tapi estuary & textile city", "Delta channels, docks, tight urban blocks."),
    "nashik": ("plateau farmland & vineyards", "Regular field geometry — where boundary recovery shows."),
    "ludhiana": ("Indo-Gangetic canal cropland", "Wheat-belt parcels on a rectilinear canal grid."),
    "nagpur": ("central Indian city & citrus country", "Compact city ringed by orchard land."),
    "coimbatore": ("Western Ghats foothills", "Textile-belt sprawl rising into the hills."),
    "ahmedabad": ("arid Sabarmati corridor", "Low-contrast dryland — a stress test for spectral fidelity."),
    "hyderabad": ("granite plateau & lakes", "Boulder fields, reservoirs, a spread-out tech corridor."),
    "jaipur": ("semi-arid Aravalli edge", "Walled city and desert fringe."),
    "bhimavaram": ("Godavari–Krishna delta", "A mosaic of aquaculture ponds — fine water boundaries everywhere."),
    "indore": ("Malwa plateau black-soil farmland", "Broad dark fields, a compact urban core."),
}

ORDER = ["vasai", "mumbai", "pune", "surat", "nashik", "ludhiana", "nagpur",
         "coimbatore", "ahmedabad", "hyderabad", "jaipur", "bhimavaram", "indore"]


def read(path, bands=None):
    with rasterio.open(path) as d:
        arr = (d.read(bands) if bands else d.read()).astype(np.float32)
        sf = float(d.tags().get("reflectance_scale_factor", 1.0) or 1.0)
    return arr * sf


def to_rgb(cube, idx, lohi):
    ch = []
    for b in idx:
        lo, hi = lohi[b]
        ch.append(np.clip((cube[b] - lo) / max(hi - lo, 1e-6), 0, 1))
    return (np.stack(ch, -1) * 255).astype(np.uint8)


def down(a, w):
    im = Image.fromarray(a)
    if im.width > w:
        im = im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
    return np.asarray(im)


def save(a, name, q):
    p = OUT / name
    Image.fromarray(a).convert("RGB").save(p, quality=q, optimize=True)
    return p.stat().st_size


def cmap_inferno(x):
    try:
        from matplotlib import cm
        return (cm.inferno(np.clip(x, 0, 1))[..., :3] * 255).astype(np.uint8)
    except Exception:
        return (np.stack([x, x * 0.55, x * 0.12], -1).clip(0, 1) * 255).astype(np.uint8)


def cmap_gate(g):
    lo, mid, hi = (np.array([0.29, 0.56, 0.77]), np.array([0.5, 0.53, 0.56]),
                   np.array([0.91, 0.58, 0.24]))
    t = g[..., None]
    out = np.where(t < 0.5, lo + (mid - lo) * (t / 0.5), mid + (hi - mid) * ((t - 0.5) / 0.5))
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def geo(path):
    with rasterio.open(path) as d:
        from rasterio.warp import transform_bounds
        b = transform_bounds(d.crs, "EPSG:4326", *d.bounds)
        return dict(lon=round((b[0] + b[2]) / 2, 3), lat=round((b[1] + b[3]) / 2, 3),
                    crs=str(d.crs))


def accuracy_from_report():
    f = EVAL / "vasai_report.json"
    b = EVAL / "vasai_boundary.json"
    out = {"ssim": "—", "psnr": "—", "psnr_b": "—",
           "boundary": "—", "boundary_b": "—", "spectral": "—", "consist": "exact"}
    step = 0
    if f.exists():
        r = json.loads(f.read_text())
        step = r.get("train_step", 0)
        d, bic = r["results"]["drishti"], r["results"]["bicubic"]
        out["ssim"] = f"{d['ssim'] * 100:.1f}%"
        out["psnr"] = f"{d['psnr_db']:.1f} dB"
        out["psnr_b"] = f"{bic['psnr_db']:.1f} dB"
        o = r.get("opensr_test") or {}
        dr, br = o.get("drishti", {}).get("reflectance"), o.get("bicubic", {}).get("reflectance")
        if dr and br:
            out["spectral"] = f"-{round((1 - dr / br) * 100)}%"
    if b.exists():
        bd = json.loads(b.read_text())
        f1d, f1b = bd["drishti_vs_truth"]["f1"], bd["bicubic_vs_truth"]["f1"]
        out["boundary_b"] = f"F1 {f1b:.2f}"
        if f1b > 0:
            out["boundary"] = f"{f1d / f1b:.1f}×"
    return out, step


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if not SRDIR.exists() or not any(SRDIR.glob("*.tif")):
        print(f"no products in {SRDIR}; run scripts/run_drishti.py per scene first", file=sys.stderr)
        return 1

    # px width / quality per layer. "deep" regions (the validated held-out
    # scene) get full-resolution imagery so you can zoom to individual 2.5 m
    # pixels; the rest stay at browse resolution to keep the file shareable.
    SIZE = {
        "deep":  {"sr": (3600, 70), "lr": (1400, 72), "nir": (1600, 68), "conf": (1800, 68), "gate": (1400, 66)},
        "browse": {"sr": (760, 68), "lr": (500, 68), "nir": (560, 66), "conf": (560, 66), "gate": (500, 64)},
    }
    DEEP = {"vasai"}

    regions, total = [], 0
    for name in ORDER:
        srp = SRDIR / f"{name}.tif"
        rawp = RAWDIR / f"{name}_s2l2a.tif"
        if not srp.exists():
            print(f"  skip {name}: {srp} missing")
            continue
        deep = name in DEEP
        z = SIZE["deep" if deep else "browse"]
        sr = read(srp)
        raw = read(rawp, [1, 2, 3, 7])
        lr_up = np.repeat(np.repeat(raw, 4, 1), 4, 2)[:, : sr.shape[1], : sr.shape[2]]
        lohi = {i: np.percentile(sr[i], (2, 98)) for i in range(4)}

        total += save(down(to_rgb(sr, [2, 1, 0], lohi), z["sr"][0]), f"{name}_sr.jpg", z["sr"][1])
        total += save(down(to_rgb(lr_up, [2, 1, 0], lohi), z["lr"][0]), f"{name}_lr.jpg", z["lr"][1])
        total += save(down(to_rgb(sr, [3, 2, 1], lohi), z["nir"][0]), f"{name}_nir.jpg", z["nir"][1])

        conf = read(srp.with_name(f"{name}_confidence.tif"))[0]
        cn = conf / max(float(np.percentile(conf, 99)), 1e-6)
        total += save(down(cmap_inferno(cn), z["conf"][0]), f"{name}_conf.jpg", z["conf"][1])
        g = read(srp.with_name(f"{name}_gate.tif"))[0] / 255.0
        total += save(down(cmap_gate(g), z["gate"][0]), f"{name}_gate.jpg", z["gate"][1])

        sidecar = json.loads(Path(str(rawp) + ".json").read_text())
        gg = geo(srp)
        ter = TERRAIN.get(name, ("", ""))
        regions.append({
            "id": name, "name": name.capitalize(),
            "lon": gg["lon"], "lat": gg["lat"], "crs": gg["crs"],
            "h10": raw.shape[1], "w10": raw.shape[2], "deep": deep,
            "date": sidecar["acquired"][:10],
            "cloud": round(float(sidecar.get("cloud_cover", 0)), 2),
            "tile": sidecar["scene_id"].split("_")[5],
            "held": name == "vasai",
            "terrain": ter[0], "note": ter[1],
            "img": {k: f"assets/{name}_{k}.jpg" for k in ("lr", "sr", "nir", "conf", "gate")},
        })
        print(f"  {name:12} {'deep' if deep else 'browse':7} ok")

    acc, step = accuracy_from_report()
    # `interim` marks imagery from a checkpoint still being trained. Set the
    # DRISHTI_FINAL env var once the run is the one you are shipping.
    interim = not os.environ.get("DRISHTI_FINAL")
    data = {"step": step or 0, "interim": interim, "accuracy": acc, "regions": regions}
    (ROOT / "frontend" / "data.js").write_text(
        "window.DRISHTI = " + json.dumps(data, indent=1) + ";\n")
    print(f"\n{len(regions)} regions · {total/1024/1024:.2f} MB assets · frontend/data.js written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
