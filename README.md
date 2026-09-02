# DRISHTI

Sentinel-2 (10 m) → 2.5 m super-resolution with a per-pixel confidence map.
SIH 2026, PS 26142. See `CLAUDE.md` for the full problem statement and the
architecture / rules the project commits to.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then add your Copernicus Data Space credentials
```

`.env` needs a Sentinel Hub OAuth client from the Copernicus Data Space
Ecosystem (create one at
<https://shapps.dataspace.copernicus.eu/dashboard/#/account/settings>):

```
CDSE_CLIENT_ID=...
CDSE_CLIENT_SECRET=...
```

## Pipeline so far (build-order steps 1–4)

### 1. Fetch a Sentinel-2 L2A scene

```bash
python scripts/fetch_aoi.py \
  --bbox 72.78 19.33 72.88 19.43 \
  --start 2025-11-01 --end 2026-02-28 --max-cloud 10 \
  --out data/raw/vasai_s2l2a.tif
```

Newest matching scene → one 10-band float32 reflectance COG on an exact 10 m
UTM grid, band order `B02 B03 B04 B05 B06 B07 B08 B8A B11 B12`, plus a
`<out>.json` sidecar (scene id, date, cloud, CRS). The 20 m bands are
delivered resampled to 10 m; DSen2-style band sharpening is a `TODO` in
`src/drishti/fetch.py`.

### 2. High-res reference

Download a LISS-IV (5.8 m) / Cartosat-1 PAN (2.5 m) patch of the same AOI from
Bhoonidhi (NRSC) and co-register it before using it for validation or
fine-tuning. *(Not yet scripted — needs a Bhoonidhi login.)*

### 3. Run the pretrained SR baseline

```bash
python scripts/run_sr.py --in data/raw/vasai_s2l2a.tif --out outputs/vasai_sr.tif --tta 4
```

Model: **SEN2SR-Lite** (`tacofoundation/sen2sr`, weights auto-downloaded to
`checkpoints/`). Deterministic x4, 10 bands, with a built-in low-frequency
hard constraint. This is a **baseline**, not the DRISHTI model.

`src/drishti/infer.py` tiles 128→512 (16 px LR overlap), runs a per-tile QA
gate *before* stitching (downsample-consistency RMSE + SAM, checked
separately on the native-10 m bands and the resampled-20 m bands), replaces
any failing tile with a bicubic upsample and records it in a QA mask, then
feather-blends the rest. Outputs next to `--out`:

| file | what |
|---|---|
| `vasai_sr.tif` | 10-band uint16 SR COG at 2.5 m (`DN = reflectance × 10000`) |
| `vasai_sr.tif.json` | fetch sidecar + SR run summary |
| `vasai_sr.tif.qa.json` | per-tile QA metrics and gate decisions |
| `vasai_sr_qa_mask.tif` | uint8: 0 = model, 1 = bicubic fallback |
| `vasai_sr_uncertainty.tif` | uint16 TTA spread (`DN = spread × 10000`; 0 = no estimate) |

The uncertainty raster here is a **test-time-augmentation proxy**. The real
MC-dropout confidence map ships with the DRISHTI model (build-order step 5).

### 4. Swipe viewer

```bash
python scripts/make_preview.py --lr data/raw/vasai_s2l2a.tif --sr outputs/vasai_sr.tif
python -m http.server -d viewer 8000   # then open http://localhost:8000
```

Self-contained, no external dependencies, runs from `file://`. Drag the handle
to swipe LR↔SR, scroll to zoom, drag to pan; toggle the uncertainty and
QA-fallback overlays; the panel shows the run's consistency metrics.

## Layout

```
src/drishti/fetch.py     Sentinel-2 L2A fetch (CDSE Process API)
src/drishti/tiling.py    tile iterator + feather-blend stitcher
src/drishti/infer.py     pretrained SR baseline + QA gate + uncertainty proxy
scripts/                 CLI wrappers
viewer/                  swipe-comparison viewer
```
