"""Run a pretrained Sentinel-2 super-resolution model over a fetched scene.

Build-order step 3: run an open pretrained SR model on our own AOI, end to end.
The model here is SEN2SR-Lite (tacofoundation/sen2sr on HuggingFace): a
deterministic 10 m -> 2.5 m (x4) model for the 10 Sentinel-2 bands with a
built-in low-frequency hard constraint, so degrade(SR) already reproduces LR.
This is a baseline, not the DRISHTI model.

Pipeline:
  1. read the multi-band reflectance COG from drishti.fetch (band order
     B02 B03 B04 B05 B06 B07 B08 B8A B11 B12 -- matches the model)
  2. tile 128x128 LR -> 512x512 SR, 16 px LR overlap
  3. per-tile QA gate BEFORE stitching: downsample-consistency RMSE + SAM
     between degrade(SR_tile) and LR_tile. Tiles that fail are filled with a
     bicubic upsample and flagged in a QA mask; they are never blended in raw.
  4. feather-blend the accepted tiles
  5. optional test-time-augmentation (D4) -> per-pixel spread as an uncertainty
     proxy. NOTE: this is not MC-dropout; the real confidence raster ships with
     the DRISHTI model. It is written so every SR product still carries an
     uncertainty layer (CLAUDE.md rule 5).

Outputs next to <out>:
  <out>                 10-band float32 reflectance COG at 2.5 m
  <out>.json            scene + run metadata sidecar
  <out>.qa.json         per-tile QA metrics and gate decisions
  <stem>_qa_mask.tif    uint8 COG, 0 = model output, 1 = bicubic fallback
  <stem>_uncertainty.tif  float32 COG, per-pixel TTA spread (if --tta > 1)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.transform import Affine

from drishti.tiling import Stitcher, iter_tiles

SR_SCALE = 4
MODEL_HF = (
    "https://huggingface.co/tacofoundation/sen2sr/resolve/main"
    "/SEN2SRLite/main/mlm.json"
)
DEFAULT_WEIGHTS_DIR = Path("checkpoints/SEN2SRLite")
BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

# Native-10 m bands vs bands delivered at 20 m and resampled to 10 m. The
# low-frequency hard constraint (CLAUDE.md rule 4) is exact on the 10 m bands
# and only approximate on the resampled ones, so the QA gate checks them with
# separate thresholds rather than lumping all 10 together.
IDX_10M = [0, 1, 2, 6]  # B02 B03 B04 B08
IDX_20M = [3, 4, 5, 7, 8, 9]  # B05 B06 B07 B8A B11 B12

# QA gate defaults. Provisional -- to be re-tuned against LISS-IV / Cartosat
# references. Gate config, not measured claims. Values chosen with margin above
# what a clean SEN2SR-Lite tile produces over Vasai (10 m: RMSE<=0.004,
# SAM<=0.65 deg; 20 m: RMSE<=0.019, SAM<=2.3 deg).
QA_RMSE_10M = 0.010
QA_SAM_10M = 1.5
QA_RMSE_20M = 0.040
QA_SAM_20M = 4.0


# --- Model loading --------------------------------------------------------


def pick_device(requested: str | None = None) -> str:
    if requested:
        return requested
    # SEN2SR-Lite is tiny; CPU has no warm-up cost and is reliable on macOS.
    return "cpu"


def load_model(
    weights_dir: str | Path = DEFAULT_WEIGHTS_DIR, device: str = "cpu"
):
    """Load SEN2SR-Lite, downloading the weights on first use."""
    import mlstac

    weights_dir = Path(weights_dir)
    if not (weights_dir / "mlm.json").exists():
        weights_dir.parent.mkdir(parents=True, exist_ok=True)
        mlstac.download(file=MODEL_HF, output_dir=str(weights_dir))
    model = mlstac.load(str(weights_dir)).compiled_model(device=device)
    return model.to(device).eval()


# --- Consistency metrics --------------------------------------------------


def degrade(sr: np.ndarray, scale: int = SR_SCALE) -> np.ndarray:
    """Block-mean an SR array (bands, H, W) down by `scale`. H, W must divide."""
    b, h, w = sr.shape
    return sr.reshape(b, h // scale, scale, w // scale, scale).mean(axis=(2, 4))


def sam_degrees(a: np.ndarray, b: np.ndarray) -> float:
    """Mean spectral angle (degrees) between two (bands, H, W) cubes."""
    av = a.reshape(a.shape[0], -1)
    bv = b.reshape(b.shape[0], -1)
    dot = np.sum(av * bv, axis=0)
    na = np.linalg.norm(av, axis=0)
    nb = np.linalg.norm(bv, axis=0)
    valid = (na > 1e-9) & (nb > 1e-9)
    cos = np.clip(dot[valid] / (na[valid] * nb[valid]), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)).mean())


# --- Inference core ------------------------------------------------------

def _forward(model, lr: np.ndarray, device: str) -> np.ndarray:
    """lr: (10, th, tw) reflectance -> sr: (10, th*4, tw*4)."""
    x = torch.from_numpy(np.ascontiguousarray(lr)).float().clamp_min(0.0)
    x = x.unsqueeze(0).to(device)
    with torch.no_grad():
        y = model(x)
    return y.squeeze(0).cpu().numpy().astype(np.float32)


def _forward_tta(
    model, lr: np.ndarray, device: str, n: int
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (mean SR, per-pixel band-mean spread) over `n` rotations (0-3
    quarter turns). The predictions are de-rotated before averaging, so the
    spread reflects how much the model's answer moves under input rotation --
    a cheap uncertainty proxy, not MC-dropout."""
    if n <= 1:
        return _forward(model, lr, device), None
    preds = []
    for k in range(min(n, 4)):
        sr_t = _forward(model, np.rot90(lr, k=k, axes=(1, 2)), device)
        preds.append(np.ascontiguousarray(np.rot90(sr_t, k=-k, axes=(1, 2))))
    stack = np.stack(preds, axis=0)
    mean = stack.mean(axis=0)
    spread = stack.std(axis=0).mean(axis=0)  # (H, W)
    return mean.astype(np.float32), spread.astype(np.float32)


def _bicubic(lr: np.ndarray, scale: int = SR_SCALE) -> np.ndarray:
    x = torch.from_numpy(np.ascontiguousarray(lr)).float().unsqueeze(0)
    y = torch.nn.functional.interpolate(
        x, scale_factor=scale, mode="bicubic", align_corners=False
    )
    return y.squeeze(0).numpy().astype(np.float32)


# --- Result / report structures ----------------------------------------


@dataclass
class TileQA:
    y0: int
    x0: int
    rmse_10m: float
    sam_10m: float
    rmse_20m: float
    sam_20m: float
    passed: bool


@dataclass
class RunReport:
    input: str
    output: str
    model: str
    device: str
    scale: int
    tile: int
    overlap: int
    tta: int
    n_tiles: int
    n_failed: int
    qa_thresholds: dict
    consistency_overall: dict  # rmse_10m, sam_10m, rmse_20m, sam_20m
    uncertainty_method: str
    run_at: str
    tiles: list[TileQA] = field(default_factory=list)


def _consistency(sr_tile: np.ndarray, lr_tile: np.ndarray) -> dict:
    """degrade(SR) vs LR, split by native-10 m vs resampled-20 m bands."""
    deg = degrade(sr_tile)
    dh, dw = deg.shape[1], deg.shape[2]
    lr_c = lr_tile[:, :dh, :dw]
    d = deg - lr_c
    return {
        "rmse_10m": float(np.sqrt(np.mean(d[IDX_10M] ** 2))),
        "sam_10m": sam_degrees(deg[IDX_10M], lr_c[IDX_10M]),
        "rmse_20m": float(np.sqrt(np.mean(d[IDX_20M] ** 2))),
        "sam_20m": sam_degrees(deg[IDX_20M], lr_c[IDX_20M]),
    }


# --- Public entry point -----------------------------------------------


REFL_SCALE = 10000  # reflectance -> uint16 DN, the Sentinel-2 L2A convention


def _quantize(refl: np.ndarray) -> np.ndarray:
    """Reflectance float -> uint16 DN (x10000), 0 reserved as nodata via +1
    guard on real zeros is unnecessary here (true 0 reflectance never occurs);
    negatives clip to 0."""
    dn = np.rint(np.clip(refl, 0.0, 6.5) * REFL_SCALE)
    return dn.astype(np.uint16)


def _write_cog(
    path: Path,
    data: np.ndarray,
    profile: dict,
    descriptions: list[str] | None = None,
    tags: dict | None = None,
    nodata: float | None = None,
) -> None:
    prof = profile.copy()
    prof.update(
        driver="COG",
        count=data.shape[0],
        dtype=data.dtype,
        compress="deflate",
        predictor=3 if np.issubdtype(data.dtype, np.floating) else 2,
        blocksize=512,
        overview_resampling="bilinear",
        BIGTIFF="IF_SAFER",
    )
    if nodata is not None:
        prof["nodata"] = nodata
    for k in ("tiled", "interleave", "blockxsize", "blockysize"):
        prof.pop(k, None)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(data)
        if descriptions:
            for i, name in enumerate(descriptions, start=1):
                dst.set_band_description(i, name)
        if tags:
            dst.update_tags(**tags)


def super_resolve(
    in_tif: str | Path,
    out_tif: str | Path,
    device: str | None = None,
    tile: int = 128,
    overlap: int = 16,
    tta: int = 4,
    qa_rmse_10m: float = QA_RMSE_10M,
    qa_sam_10m: float = QA_SAM_10M,
    qa_rmse_20m: float = QA_RMSE_20M,
    qa_sam_20m: float = QA_SAM_20M,
    weights_dir: str | Path = DEFAULT_WEIGHTS_DIR,
) -> RunReport:
    in_tif, out_tif = Path(in_tif), Path(out_tif)
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    device = pick_device(device)
    model = load_model(weights_dir, device)

    with rasterio.open(in_tif) as src:
        lr = src.read().astype(np.float32)  # (10, H, W)
        src_profile = src.profile
        src_transform = src.transform
        src_crs = src.crs
        descriptions = src.descriptions

    if lr.shape[0] != len(BANDS):
        raise ValueError(
            f"expected {len(BANDS)} bands in {in_tif}, got {lr.shape[0]}"
        )
    _, H, W = lr.shape
    sr_h, sr_w = H * SR_SCALE, W * SR_SCALE

    tiles = iter_tiles(H, W, tile, overlap)
    stitch = Stitcher(len(BANDS), sr_h, sr_w, SR_SCALE, ramp=overlap)
    unc_stitch = (
        Stitcher(1, sr_h, sr_w, SR_SCALE, ramp=overlap, allow_gaps=True)
        if tta > 1
        else None
    )
    qa_mask = np.zeros((sr_h, sr_w), dtype=np.uint8)

    thresholds = {
        "rmse_10m": qa_rmse_10m,
        "sam_10m": qa_sam_10m,
        "rmse_20m": qa_rmse_20m,
        "sam_20m": qa_sam_20m,
    }
    tile_qa: list[TileQA] = []
    passed_metrics: list[dict] = []

    for t in tiles:
        lr_tile = lr[:, t.y0 : t.y0 + t.h, t.x0 : t.x0 + t.w]
        sr_tile, spread = _forward_tta(model, lr_tile, device, tta)
        sr_tile = np.clip(sr_tile, 0.0, None)

        m = _consistency(sr_tile, lr_tile)
        passed = (
            m["rmse_10m"] <= qa_rmse_10m
            and m["sam_10m"] <= qa_sam_10m
            and m["rmse_20m"] <= qa_rmse_20m
            and m["sam_20m"] <= qa_sam_20m
        )
        tile_qa.append(
            TileQA(
                t.y0,
                t.x0,
                round(m["rmse_10m"], 6),
                round(m["sam_10m"], 4),
                round(m["rmse_20m"], 6),
                round(m["sam_20m"], 4),
                passed,
            )
        )

        if not passed:
            sr_tile = np.clip(_bicubic(lr_tile), 0.0, None)
            sy, sx = t.y0 * SR_SCALE, t.x0 * SR_SCALE
            qa_mask[sy : sy + sr_tile.shape[1], sx : sx + sr_tile.shape[2]] = 1
            spread = None
        else:
            passed_metrics.append(m)

        stitch.add(t, sr_tile)
        if unc_stitch is not None and spread is not None:
            unc_stitch.add(t, spread[None])

    sr = stitch.result()

    def _mean(key: str) -> float:
        vals = [d[key] for d in passed_metrics]
        return round(float(np.mean(vals)), 6) if vals else float("nan")

    overall = {k: _mean(k) for k in thresholds}

    dst_transform = Affine(
        src_transform.a / SR_SCALE,
        src_transform.b,
        src_transform.c,
        src_transform.d,
        src_transform.e / SR_SCALE,
        src_transform.f,
    )
    base_profile = {
        "height": sr_h,
        "width": sr_w,
        "crs": src_crs,
        "transform": dst_transform,
    }

    band_names = [
        (descriptions[i] if descriptions and descriptions[i] else BANDS[i])
        for i in range(len(BANDS))
    ]
    _write_cog(
        out_tif,
        _quantize(sr),
        base_profile,
        descriptions=band_names,
        nodata=0,
        tags={
            "model": "SEN2SR-Lite",
            "scale": str(SR_SCALE),
            "reflectance_scale_factor": str(1.0 / REFL_SCALE),
            "reflectance_units": f"DN = reflectance * {REFL_SCALE} (uint16)",
            "source": "DRISHTI drishti.infer",
        },
    )

    stem = out_tif.with_suffix("")
    _write_cog(
        Path(f"{stem}_qa_mask.tif"),
        qa_mask[None],
        base_profile,
        descriptions=["qa_mask (0=model, 1=bicubic fallback)"],
    )
    if unc_stitch is not None:
        unc = unc_stitch.result()  # reflectance-scale spread, NaN in gaps
        unc_dn = np.where(
            np.isfinite(unc), np.clip(unc, 0, 6.5) * REFL_SCALE, 0
        )
        _write_cog(
            Path(f"{stem}_uncertainty.tif"),
            np.rint(unc_dn).astype(np.uint16),
            base_profile,
            descriptions=["tta_spread (DN = spread * 10000; 0 = no estimate)"],
            nodata=0,
        )

    report = RunReport(
        input=str(in_tif),
        output=str(out_tif),
        model="SEN2SR-Lite (tacofoundation/sen2sr)",
        device=device,
        scale=SR_SCALE,
        tile=tile,
        overlap=overlap,
        tta=tta,
        n_tiles=len(tiles),
        n_failed=sum(1 for q in tile_qa if not q.passed),
        qa_thresholds=thresholds,
        consistency_overall=overall,
        uncertainty_method=(
            "test-time-augmentation spread (D4, n={}); proxy only -- "
            "MC-dropout confidence ships with the DRISHTI model".format(tta)
            if tta > 1
            else "none (tta<=1)"
        ),
        run_at=datetime.now(timezone.utc).isoformat(),
        tiles=tile_qa,
    )
    Path(f"{out_tif}.qa.json").write_text(
        json.dumps(asdict(report), indent=2)
    )

    # Carry the fetch sidecar forward, annotated with the SR step.
    src_sidecar = Path(f"{in_tif}.json")
    sidecar: dict = {}
    if src_sidecar.exists():
        sidecar = json.loads(src_sidecar.read_text())
    sidecar["super_resolution"] = {
        "model": report.model,
        "scale": SR_SCALE,
        "resolution_m": sidecar.get("resolution_m", 10.0) / SR_SCALE,
        "consistency_overall": report.consistency_overall,
        "n_tiles": report.n_tiles,
        "n_failed_tiles": report.n_failed,
        "uncertainty_method": report.uncertainty_method,
        "run_at": report.run_at,
    }
    Path(f"{out_tif}.json").write_text(json.dumps(sidecar, indent=2))

    return report
