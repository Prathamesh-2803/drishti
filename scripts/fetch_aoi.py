#!/usr/bin/env python3
"""CLI wrapper around drishti.fetch.fetch_scene.

Example:
    python scripts/fetch_aoi.py \
        --bbox 72.80 18.90 72.90 19.00 \
        --start 2024-11-01 --end 2024-12-31 \
        --max-cloud 10 \
        --out data/raw/mumbai_s2l2a.tif
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make src/ importable when run from the repo root without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drishti.fetch import fetch_scene  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(
        description="Fetch the newest matching Sentinel-2 L2A scene as a "
        "multi-band Cloud-Optimised GeoTIFF in surface reflectance."
    )
    p.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        required=True,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Bounding box in EPSG:4326 degrees.",
    )
    p.add_argument("--start", required=True, help="Start date, YYYY-MM-DD (UTC).")
    p.add_argument("--end", required=True, help="End date, YYYY-MM-DD (UTC).")
    p.add_argument(
        "--max-cloud",
        type=float,
        default=10.0,
        help="Maximum scene cloud cover in percent (default: 10).",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output GeoTIFF path. A '<out>.json' sidecar is written alongside.",
    )
    args = p.parse_args()

    try:
        scene = fetch_scene(
            bbox=args.bbox,
            start=args.start,
            end=args.end,
            max_cloud=args.max_cloud,
            out=args.out,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Scene:      {scene.scene_id}")
    print(f"Acquired:   {scene.acquired}")
    print(f"Cloud:      {scene.cloud_cover}%")
    print(f"CRS:        {scene.crs}")
    print(f"Bands:      {', '.join(scene.bands)}")
    print(f"Written:    {args.out}")
    print(f"Sidecar:    {args.out}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
