#!/usr/bin/env python3
"""Run the trained DRISHTI model on a real Sentinel-2 scene: 10 m -> 2.5 m.

This is the *product* path (build-order step 5). It is distinct from
scripts/evaluate.py, which *measures* quality under Wald's protocol on a held-out
scene. Here there is no 2.5 m ground truth: the real 10 m scene is super-resolved
at the true 4x, and every output ships with its confidence raster and the gate
map (CLAUDE.md rule 5 -- the confidence raster is a product, not a debug artifact).

    python scripts/run_drishti.py --ckpt checkpoints/drishti/best.pt \
        --scene data/raw/vasai_s2l2a.tif --out outputs/vasai_drishti_sr.tif

Outputs next to <out>:
    <out>                   4-band uint16 reflectance COG at 2.5 m (B02 B03 B04 B08)
    <stem>_confidence.tif   uint16 MC-dropout std, DN = std * 10000
    <stem>_gate.tif         uint8 mean gate (0..255; 255 = CNN branch fully trusted)
    <stem>_qa_mask.tif      uint8: 0 = passed, 1 = flagged (high uncertainty / low
                            downsample-consistency). Flag only -- tiles are never
                            dropped because the hard constraint already bounds them.
    <out>.json              fetch sidecar + DRISHTI run summary
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.transform import Affine

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drishti.data import RGBN, load_scene  # noqa: E402
from drishti.infer import REFL_SCALE, _write_cog  # noqa: E402
from drishti.metrics import sam_degrees  # noqa: E402
from drishti.model.drishti import Drishti  # noqa: E402
from drishti.tiling import Stitcher, iter_tiles  # noqa: E402

# Independent downsample-consistency check for the QA gate. The model trains with
# a box filter (sigma=0); here we degrade with a sensor-like blur so the check is
# not circular with the hard constraint. Provisional thresholds -- gate config,
# not measured claims.
QA_BLUR_SIGMA = 1.6            # 0.4 * scale
QA_CONS_RMSE = 0.02           # reflectance
QA_CONS_SAM = 2.5             # degrees
QA_CONF_P95 = 0.05            # reflectance; tile flagged if its p95 confidence exceeds this


def pick_device(req: str) -> torch.device:
    if req != "auto":
        return torch.device(req)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _blur_degrade(x: torch.Tensor, scale: int, sigma: float) -> torch.Tensor:
    from drishti.data import degrade

    return degrade(x, scale, sigma)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--scene", required=True, help="real 10 m Sentinel-2 L2A COG")
    p.add_argument("--out", required=True, help="output SR COG path")
    p.add_argument("--tile", type=int, default=128, help="LR tile size (px)")
    p.add_argument("--overlap", type=int, default=16, help="LR tile overlap (px)")
    p.add_argument("--passes", type=int, default=16, help="MC-dropout passes for confidence")
    p.add_argument("--max-size", type=int, default=0,
                   help="if >0, center-crop the scene to at most this many px a side")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    dev = pick_device(args.device)
    out_tif = Path(args.out)
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    stem = out_tif.with_suffix("")

    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    a = ck.get("args", {})
    scale = a.get("scale", 4)
    model = Drishti(bands=len(RGBN), ch=a.get("ch", 64), cnn_blocks=a.get("cnn_blocks", 8),
                    swin_depth=a.get("swin_depth", 4), window=a.get("window", 8),
                    scale=scale, sigma=a.get("sigma", 0.0),
                    dropout=a.get("dropout", 0.1)).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    sc = load_scene(args.scene, RGBN)
    with rasterio.open(args.scene) as ds:
        src_transform, src_crs = ds.transform, ds.crs
    arr = sc.array
    y_off = x_off = 0
    if args.max_size and (arr.shape[1] > args.max_size or arr.shape[2] > args.max_size):
        y_off = (arr.shape[1] - args.max_size) // 2
        x_off = (arr.shape[2] - args.max_size) // 2
        arr = arr[:, y_off:y_off + args.max_size, x_off:x_off + args.max_size]
    _, H, W = arr.shape
    lr = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0)

    sr_h, sr_w = H * scale, W * scale
    st_sr = Stitcher(len(RGBN), sr_h, sr_w, scale, ramp=args.overlap)
    st_cf = Stitcher(1, sr_h, sr_w, scale, ramp=args.overlap)
    st_gt = Stitcher(1, sr_h, sr_w, scale, ramp=args.overlap)
    qa_mask = np.zeros((sr_h, sr_w), dtype=np.uint8)

    tiles = iter_tiles(H, W, args.tile, args.overlap)
    print(f"scene   : {sc.name}  {H}x{W} px 10 m  ->  {sr_h}x{sr_w} px 2.5 m")
    print(f"model   : {args.ckpt} (step {ck.get('step')})  device {dev}  {len(tiles)} tiles")

    qa_rows, n_flag = [], 0
    for i, t in enumerate(tiles):
        patch = lr[:, :, t.y0:t.y0 + t.h, t.x0:t.x0 + t.w].to(dev)
        with torch.no_grad():
            mean, std = model.predict_with_uncertainty(patch, passes=args.passes)
            _, gate = model(patch, return_gate=True)
        sr_t = mean[0].cpu().numpy()
        cf_t = std[0].cpu().numpy()

        deg = _blur_degrade(mean, scale, QA_BLUR_SIGMA)
        cons_rmse = float((deg - patch).pow(2).mean().sqrt())
        cons_sam = sam_degrees(deg, patch)
        conf_p95 = float(np.percentile(cf_t, 95))
        passed = (cons_rmse <= QA_CONS_RMSE and cons_sam <= QA_CONS_SAM
                  and conf_p95 <= QA_CONF_P95)
        qa_rows.append({"y0": t.y0, "x0": t.x0, "cons_rmse": round(cons_rmse, 6),
                        "cons_sam_deg": round(cons_sam, 4), "conf_p95": round(conf_p95, 6),
                        "passed": passed})
        if not passed:
            n_flag += 1
            sy, sx = t.y0 * scale, t.x0 * scale
            qa_mask[sy:sy + sr_t.shape[1], sx:sx + sr_t.shape[2]] = 1

        st_sr.add(t, sr_t)
        st_cf.add(t, cf_t)
        # gate lives on the LR grid -> nearest-up to the SR grid for display
        g = np.repeat(np.repeat(gate[0].cpu().numpy(), scale, 1), scale, 2)
        st_gt.add(t, g)
        if (i + 1) % 20 == 0 or i + 1 == len(tiles):
            print(f"  tile {i+1}/{len(tiles)}", flush=True)

    sr = np.clip(st_sr.result(), 0.0, None)          # (bands, H, W)
    conf = np.clip(st_cf.result(), 0.0, None)         # (1, H, W)
    gate_img = np.clip(st_gt.result(), 0.0, 1.0)      # (1, H, W)

    dst_transform = Affine(src_transform.a / scale, src_transform.b,
                           src_transform.c + x_off * src_transform.a,
                           src_transform.d, src_transform.e / scale,
                           src_transform.f + y_off * src_transform.e)
    base = {"height": sr_h, "width": sr_w, "crs": src_crs, "transform": dst_transform}

    dn = np.rint(np.clip(sr, 0.0, 6.5) * REFL_SCALE).astype(np.uint16)
    _write_cog(out_tif, dn, base, descriptions=RGBN, nodata=0, tags={
        "model": "DRISHTI", "scale": str(scale),
        "reflectance_scale_factor": str(1.0 / REFL_SCALE),
        "checkpoint": str(args.ckpt), "train_step": str(ck.get("step")),
        "source": "DRISHTI scripts/run_drishti.py"})

    conf_dn = np.rint(np.clip(conf, 0.0, 6.5) * REFL_SCALE).astype(np.uint16)
    _write_cog(Path(f"{stem}_confidence.tif"), conf_dn, base,
               descriptions=[f"mc_dropout_std (DN = std * {REFL_SCALE}); {args.passes} passes"],
               nodata=0)
    _write_cog(Path(f"{stem}_gate.tif"), np.rint(gate_img * 255).astype(np.uint8), base,
               descriptions=["mean gate (255 = CNN branch fully trusted, 0 = Swin branch)"])
    _write_cog(Path(f"{stem}_qa_mask.tif"), qa_mask[None], base,
               descriptions=["qa_mask (0 = passed, 1 = flagged: high uncertainty / low consistency)"])

    sidecar_path = Path(f"{args.scene}.json")
    sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.exists() else {}
    sidecar["drishti_sr"] = {
        "model": "DRISHTI (hybrid CNN + Swin, learned gate, exact low-freq constraint)",
        "checkpoint": str(args.ckpt),
        "train_step": ck.get("step"),
        "scale": scale,
        "resolution_m": sidecar.get("resolution_m", 10.0) / scale,
        "bands": RGBN,
        "mc_dropout_passes": args.passes,
        "confidence_mean_reflectance": round(float(conf.mean()), 6),
        "confidence_p95_reflectance": round(float(np.percentile(conf, 95)), 6),
        "gate_mean": round(float(gate_img.mean()), 4),
        "n_tiles": len(tiles),
        "n_tiles_flagged": n_flag,
        "qa_thresholds": {"cons_rmse": QA_CONS_RMSE, "cons_sam_deg": QA_CONS_SAM,
                          "conf_p95": QA_CONF_P95, "blur_sigma": QA_BLUR_SIGMA},
        "run_at": datetime.now(timezone.utc).isoformat(),
    }
    Path(f"{args.out}.json").write_text(json.dumps(sidecar, indent=2))
    Path(f"{args.out}.qa.json").write_text(json.dumps(
        {"scene": sc.name, "n_tiles": len(tiles), "n_flagged": n_flag, "tiles": qa_rows},
        indent=2))

    # block-average consistency guarantee, measured on the whole scene
    sr_t = torch.from_numpy(sr).unsqueeze(0)
    blockavg = torch.nn.functional.avg_pool2d(sr_t, scale)
    ba_rmse = float((blockavg - lr).pow(2).mean().sqrt())
    print(f"\nblock-average consistency RMSE (SR -> 10 m vs input): {ba_rmse:.2e}")
    print(f"tiles flagged: {n_flag}/{len(tiles)}   confidence mean {conf.mean():.5f}  "
          f"p95 {np.percentile(conf, 95):.5f}")
    print(f"written: {out_tif} (+ _confidence, _gate, _qa_mask, .json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
