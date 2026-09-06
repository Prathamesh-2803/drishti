#!/usr/bin/env bash
# Fetch extra Sentinel-2 L2A scenes across varied Indian terrain for training
# diversity. Sequential, so we stay gentle on the CDSE API. Skips any scene
# whose output already exists.
set -u
cd "$(dirname "$0")/.."

START=2024-11-01
END=2026-03-01
CLOUD=5

# name              min_lon min_lat max_lon max_lat
AOIS=(
  "ludhiana_s2l2a   75.80 30.85 75.90 30.95"
  "nagpur_s2l2a     79.05 21.10 79.15 21.20"
  "coimbatore_s2l2a 76.95 11.00 77.05 11.10"
  "ahmedabad_s2l2a  72.55 23.00 72.65 23.10"
  "hyderabad_s2l2a  78.45 17.40 78.55 17.50"
  "jaipur_s2l2a     75.75 26.90 75.85 27.00"
  "bhimavaram_s2l2a 81.50 16.50 81.60 16.60"
  "indore_s2l2a     75.85 22.70 75.95 22.80"
)

for row in "${AOIS[@]}"; do
  read -r name lon0 lat0 lon1 lat1 <<< "$row"
  out="data/raw/${name}.tif"
  if [[ -f "$out" ]]; then
    echo "== $name: exists, skip"
    continue
  fi
  echo "== $name: fetching $lon0 $lat0 $lon1 $lat1"
  python scripts/fetch_aoi.py --bbox "$lon0" "$lat0" "$lon1" "$lat1" \
    --start "$START" --end "$END" --max-cloud "$CLOUD" --out "$out" \
    || echo "!! $name FAILED"
done
echo "== batch fetch done"
