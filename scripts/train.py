#!/usr/bin/env python3
"""Train DRISHTI under Wald's protocol.

    python scripts/train.py \
        --train data/raw/vasai_s2l2a.tif data/raw/mumbai_s2l2a.tif \
                data/raw/nashik_s2l2a.tif data/raw/pune_s2l2a.tif \
        --holdout data/raw/surat_s2l2a.tif \
        --steps 6000 --batch 8 --out checkpoints/drishti

The held-out scene is never sampled for training. It is used only to log
validation numbers during the run, and the final report is produced separately
by scripts/evaluate.py on that same untouched scene.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drishti.data import (RGBN, WaldPatches, degrade, load_scene, split_scene,  # noqa: E402
                          upsample)
from drishti.metrics import DrishtiLoss, evaluate_all  # noqa: E402
from drishti.model.drishti import Drishti, count_params  # noqa: E402


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def val_crop(scene, size: int, scale: int, sigma: float | None):
    """A fixed centre crop of the held-out region, so validation is comparable
    between steps rather than drifting with a random sample."""
    import numpy as np

    _, H, W = scene.array.shape
    size = min(size, H - H % scale, W - W % scale)
    y, x = (H - size) // 2, (W - size) // 2
    crop = scene.array[:, y : y + size, x : x + size]
    hr = torch.from_numpy(np.ascontiguousarray(crop)).unsqueeze(0)
    return degrade(hr, scale, sigma), hr


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--train", nargs="+", required=True, help="training scene GeoTIFFs")
    p.add_argument("--holdout", default=None,
                   help="held-out scene GeoTIFF, never trained on (cross-region split)")
    p.add_argument("--spatial-holdout", type=float, default=None, metavar="FRAC",
                   help="instead of a separate scene, reserve the southern FRAC of every "
                        "training scene. Use when only one region is available.")
    p.add_argument("--gap", type=int, default=64,
                   help="rows discarded between the training and held-out regions, so "
                        "spatial autocorrelation cannot leak across the boundary")
    p.add_argument("--out", default="checkpoints/drishti")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--hr-patch", type=int, default=256)
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--sigma", type=float, default=None, help="degradation blur; default 0.4*scale")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ch", type=int, default=64)
    p.add_argument("--cnn-blocks", type=int, default=8)
    p.add_argument("--swin-depth", type=int, default=4)
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--w-sam", type=float, default=0.15)
    p.add_argument("--w-cons", type=float, default=0.5)
    p.add_argument("--w-grad", type=float, default=0.0,
                   help="edge-map (Sobel) loss weight; a late low-weight sharpness "
                        "term, e.g. 0.05 for a perceptual fine-tune")
    p.add_argument("--val-every", type=int, default=250)
    p.add_argument("--val-size", type=int, default=512)
    p.add_argument("--device", default="auto")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", default=None,
                   help="resume a run: restores model + optimizer + schedule + step")
    p.add_argument("--init-from", default=None,
                   help="load only model weights from this checkpoint and start a fresh "
                        "schedule at step 0 (for a fine-tune)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if (args.holdout is None) == (args.spatial_holdout is None):
        raise SystemExit("give exactly one of --holdout or --spatial-holdout")

    loaded = [load_scene(t, RGBN) for t in args.train]
    if args.holdout is not None:
        holdout_name = Path(args.holdout).stem
        if holdout_name in [Path(t).stem for t in args.train]:
            raise SystemExit("refusing to run: the held-out scene is also in --train")
        scenes = loaded
        hold_scene = load_scene(args.holdout, RGBN)
        split_note = f"separate region ({hold_scene.name})"
    else:
        pairs = [split_scene(s, args.spatial_holdout, args.gap) for s in loaded]
        scenes = [tr for tr, _ in pairs]
        hold_scene = pairs[0][1]
        split_note = (f"southern {args.spatial_holdout:.0%} of each scene, "
                      f"{args.gap}-row gap discarded")
    holdout_name = hold_scene.name

    print(f"device      : {device}")
    print(f"train scenes: {', '.join(s.name for s in scenes)}")
    print(f"held out    : {holdout_name}  ({split_note}) -- never sampled")
    for s in scenes:
        print(f"  train  {s.name:<18} {s.array.shape}")
    print(f"  hold   {hold_scene.name:<18} {hold_scene.array.shape}")

    ds = WaldPatches(scenes, hr_patch=args.hr_patch, scale=args.scale,
                     length=args.steps * args.batch, sigma=args.sigma, seed=args.seed)
    dl = DataLoader(ds, batch_size=args.batch, num_workers=args.workers,
                    shuffle=False, drop_last=True, pin_memory=(device.type == "cuda"))

    model = Drishti(bands=len(RGBN), ch=args.ch, cnn_blocks=args.cnn_blocks,
                    swin_depth=args.swin_depth, window=args.window,
                    scale=args.scale, sigma=args.sigma, dropout=args.dropout).to(device)
    print(f"parameters  : {count_params(model):,}")

    if args.init_from:
        ick = torch.load(args.init_from, map_location=device, weights_only=False)
        model.load_state_dict(ick["model"])
        print(f"init weights: {args.init_from} (step {ick.get('step')}) -- fresh schedule")

    crit = DrishtiLoss(w_sam=args.w_sam, w_cons=args.w_cons, w_grad=args.w_grad,
                       scale=args.scale, sigma=args.sigma)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.05)

    start_step = 0
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); start_step = ck["step"]
        print(f"resumed from {args.resume} at step {start_step}")

    vlr, vhr = val_crop(hold_scene, args.val_size, args.scale, args.sigma)
    vlr, vhr = vlr.to(device), vhr.to(device)

    log_path = out_dir / "train_log.csv"
    new_log = not log_path.exists()
    log = open(log_path, "a", newline="")
    writer = csv.writer(log)
    if new_log:
        writer.writerow(["step", "lr", "loss", "l1", "sam_rad", "cons",
                         "val_psnr_db", "val_ssim", "val_sam_deg", "val_ndvi_mae",
                         "bicubic_psnr_db", "secs"])

    best = -1.0
    t0 = time.time()
    model.train()
    running = {"total": 0.0, "l1": 0.0, "sam_rad": 0.0, "cons": 0.0}
    if args.w_grad:
        running["grad"] = 0.0
    n_acc = 0

    for step, (lr_b, hr_b) in enumerate(dl, start=start_step + 1):
        lr_b, hr_b = lr_b.to(device), hr_b.to(device)
        sr = model(lr_b)
        loss, parts = crit(sr, hr_b, lr_b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        for k in running:
            running[k] += parts.get(k, 0.0)
        n_acc += 1

        if step % 50 == 0:
            msg = " ".join(f"{k}={running[k]/n_acc:.5f}" for k in running)
            print(f"step {step:>6}/{args.steps}  lr={sched.get_last_lr()[0]:.2e}  {msg}"
                  f"  ({time.time()-t0:.0f}s)", flush=True)

        if step % args.val_every == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                vsr = model(vlr)
                m = evaluate_all(vsr, vhr, vlr, args.scale, args.sigma)
                bic = evaluate_all(upsample(vlr, args.scale), vhr, vlr, args.scale, args.sigma)
            model.train()
            print(f"  [val {holdout_name}] PSNR {m['psnr_db']:.3f} dB "
                  f"(bicubic {bic['psnr_db']:.3f})  SSIM {m['ssim']:.4f}  "
                  f"SAM {m['sam_deg']:.4f} deg  NDVI MAE {m['ndvi_mae']:.5f}", flush=True)
            writer.writerow([step, f"{sched.get_last_lr()[0]:.6e}",
                             f"{running['total']/max(n_acc,1):.6f}",
                             f"{running['l1']/max(n_acc,1):.6f}",
                             f"{running['sam_rad']/max(n_acc,1):.6f}",
                             f"{running['cons']/max(n_acc,1):.6f}",
                             f"{m['psnr_db']:.4f}", f"{m['ssim']:.5f}",
                             f"{m['sam_deg']:.5f}", f"{m['ndvi_mae']:.6f}",
                             f"{bic['psnr_db']:.4f}", f"{time.time()-t0:.0f}"])
            log.flush()
            running = {k: 0.0 for k in running}; n_acc = 0

            ck = {"model": model.state_dict(), "opt": opt.state_dict(),
                  "sched": sched.state_dict(), "step": step, "args": vars(args)}
            torch.save(ck, out_dir / "last.pt")
            if m["psnr_db"] > best:
                best = m["psnr_db"]
                torch.save(ck, out_dir / "best.pt")
                (out_dir / "best.json").write_text(json.dumps(
                    {"step": step, "val_scene": holdout_name, "split": split_note, **m,
                     "bicubic_psnr_db": bic["psnr_db"]}, indent=2))

        if step >= args.steps:
            break

    log.close()
    print(f"\ndone in {time.time()-t0:.0f}s. best val PSNR {best:.3f} dB -> {out_dir/'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
