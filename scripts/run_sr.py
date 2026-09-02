#!/usr/bin/env python3
"""CLI: run the pretrained SEN2SR-Lite baseline over a fetched Sentinel-2 COG.

Example:
    python scripts/run_sr.py \
        --in  data/raw/vasai_s2l2a.tif \
        --out outputs/vasai_sr.tif \
        --tta 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drishti.infer import super_resolve  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="in_tif", required=True, help="Input LR COG.")
    p.add_argument("--out", dest="out_tif", required=True, help="Output SR COG.")
    p.add_argument(
        "--device",
        default=None,
        help="torch device (cpu / mps / cuda). Default: cpu.",
    )
    p.add_argument("--tile", type=int, default=128, help="LR tile size (px).")
    p.add_argument("--overlap", type=int, default=16, help="LR tile overlap (px).")
    p.add_argument(
        "--tta",
        type=int,
        default=4,
        help="Test-time-augmentation passes (1 = off) for the uncertainty "
        "proxy. Max 4.",
    )
    p.add_argument("--qa-rmse-10m", type=float, default=None)
    p.add_argument("--qa-sam-10m", type=float, default=None)
    p.add_argument("--qa-rmse-20m", type=float, default=None)
    p.add_argument("--qa-sam-20m", type=float, default=None)
    args = p.parse_args()

    from drishti.infer import (  # noqa: E402
        QA_RMSE_10M,
        QA_RMSE_20M,
        QA_SAM_10M,
        QA_SAM_20M,
    )

    try:
        r = super_resolve(
            in_tif=args.in_tif,
            out_tif=args.out_tif,
            device=args.device,
            tile=args.tile,
            overlap=args.overlap,
            tta=max(1, min(4, args.tta)),
            qa_rmse_10m=args.qa_rmse_10m or QA_RMSE_10M,
            qa_sam_10m=args.qa_sam_10m or QA_SAM_10M,
            qa_rmse_20m=args.qa_rmse_20m or QA_RMSE_20M,
            qa_sam_20m=args.qa_sam_20m or QA_SAM_20M,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    c = r.consistency_overall
    print(f"Model:       {r.model} on {r.device}")
    print(f"Tiles:       {r.n_tiles}  ({r.n_failed} failed QA -> bicubic)")
    print(
        f"Consistency: degrade(SR) vs LR (passed tiles mean)  "
        f"10 m bands RMSE {c['rmse_10m']} / SAM {c['sam_10m']}deg  |  "
        f"20 m bands RMSE {c['rmse_20m']} / SAM {c['sam_20m']}deg"
    )
    print(f"Uncertainty: {r.uncertainty_method}")
    print(f"Written:     {r.output}")
    print(f"             {r.output}.json  {r.output}.qa.json")
    stem = Path(r.output).with_suffix("")
    print(f"             {stem}_qa_mask.tif")
    if r.tta > 1:
        print(f"             {stem}_uncertainty.tif")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
