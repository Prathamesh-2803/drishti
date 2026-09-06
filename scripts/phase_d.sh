#!/usr/bin/env bash
# Phase D — everything that turns a trained checkpoint into the deliverables:
#   1. measured evaluation on the held-out scene (vs bicubic + opensr-test)
#   2. downstream field-boundary task
#   3. run the model over all 13 scenes -> 2.5 m products + confidence + gate
#   4. build the frontend web assets + data.js
#
#   scripts/phase_d.sh [checkpoints/drishti/best.pt]
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT=${1:-checkpoints/drishti/best.pt}
HELD=vasai
EVAL=outputs/eval_vasai
SRDIR=outputs/sr
DEV=${DEV:-mps}
PASSES=${PASSES:-16}

SCENES=(vasai mumbai pune surat nashik ludhiana nagpur coimbatore ahmedabad hyderabad jaipur bhimavaram indore)

echo "== 1/4  measured evaluation — held-out $HELD"
python scripts/evaluate.py --ckpt "$CKPT" --scene "data/raw/${HELD}_s2l2a.tif" \
  --out "$EVAL" --max-size 1024 --passes "$PASSES" --device "$DEV"

echo "== 2/4  downstream field-boundary task"
python scripts/downstream_boundary.py --ckpt "$CKPT" --scene "data/raw/${HELD}_s2l2a.tif" \
  --out "$EVAL" --max-size 1024 --canny-lo 0.14 --canny-hi 0.28 --device "$DEV"

echo "== 3/4  products over all ${#SCENES[@]} scenes"
mkdir -p "$SRDIR"
for s in "${SCENES[@]}"; do
  echo "   -- $s"
  python scripts/run_drishti.py --ckpt "$CKPT" --scene "data/raw/${s}_s2l2a.tif" \
    --out "$SRDIR/${s}.tif" --passes "$PASSES" --device "$DEV"
done

echo "== 4/4  frontend assets + data.js"
python scripts/build_frontend_assets.py

echo
echo "== measured numbers =="
cat "$EVAL/vasai_report.md"
echo; cat "$EVAL/vasai_boundary.json"
