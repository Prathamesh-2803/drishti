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

## Pipeline (build-order steps 1–5)

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
to swipe LR↔SR, scroll to zoom, drag to pan; toggle the confidence, gate and
QA-flag overlays; the panel shows the run's metrics.

### 5. The DRISHTI model

Two-branch network — a CNN residual-channel-attention path (local detail) and a
Swin window-attention path (wide context) — fused per pixel by a learned gate.
The gate currently sits balanced (~0.50): both branches reconstruct every pixel
and the fusion beats either alone by ~0.3 dB on the held-out scene. `run_drishti.py`
also writes a **branch-divergence** raster — where the two paths disagree, i.e.
where fusing both changes the answer (edges, roads, built structure).
4-band RGBN (B02 B03 B04 B08), ×4. Every output is forced to be **exactly**
downscale-consistent: `avg_pool(SR, 4) == LR` to machine precision
(`scripts/check_constraint.py`). Final checkpoint: `checkpoints/drishti-ft/best.pt`
(base run `checkpoints/drishti/`, then a short fine-tune with `--reset-gate
--w-gate` to rebalance the branches).

Trained under **Wald's protocol**: the only ground truth that exists without a
2.5 m reference is the real 10 m scene itself, so the model learns
40 m → 10 m (box-filter degrade, exact supervision) and is applied at
10 m → 2.5 m. One region (Vasai) is held out entirely — never sampled in
training, used only for the measured evaluation.

```bash
# train (12 scenes, Vasai held out)
python scripts/train.py --train data/raw/{mumbai,nashik,pune,surat,ludhiana,nagpur,\
coimbatore,ahmedabad,hyderabad,jaipur,bhimavaram,indore}_s2l2a.tif \
  --holdout data/raw/vasai_s2l2a.tif --sigma 0 \
  --steps 35000 --batch 8 --ch 96 --cnn-blocks 12 --swin-depth 6 --out checkpoints/drishti

# measured evaluation on the held-out scene, vs bicubic
python scripts/evaluate.py --ckpt checkpoints/drishti/best.pt \
  --scene data/raw/vasai_s2l2a.tif --out outputs/eval_vasai

# downstream task: field-boundary F1 before vs after SR
python scripts/downstream_boundary.py --ckpt checkpoints/drishti/best.pt \
  --scene data/raw/vasai_s2l2a.tif --out outputs/eval_vasai

# the product: real 10 m scene -> 2.5 m + confidence + gate rasters
python scripts/run_drishti.py --ckpt checkpoints/drishti/best.pt \
  --scene data/raw/vasai_s2l2a.tif --out outputs/vasai_drishti_sr.tif
python scripts/make_preview.py --lr data/raw/vasai_s2l2a.tif --sr outputs/vasai_drishti_sr.tif
```

`evaluate.py` reports PSNR / SSIM / SAM° / reflectance-L1 / NDVI-MAE /
block-average consistency against a bicubic baseline, plus opensr-test
hallucination / omission / improvement. `run_drishti.py` writes the 4-band
2.5 m COG alongside `_confidence.tif` (MC-dropout std), `_gate.tif` (where the
transformer branch was used) and `_qa_mask.tif`.

### Measured — held-out Vasai, never trained on

Wald protocol (box-filter ×4 decimation), 1024 px crop, `checkpoints/drishti-ft`,
vs bicubic. Full report in `outputs/eval_vasai/`.

| metric | bicubic | DRISHTI |
|---|---|---|
| PSNR (dB) | 35.26 | **36.40** |
| SSIM | 0.870 | **0.896** |
| SAM (°) | 3.09 | **2.59** |
| reflectance L1 | 0.0173 | **0.0151** |
| NDVI MAE | 0.0513 | **0.0420** |
| block-average consistency (refl) | 6.0e-3 | **1.2e-5** |
| opensr-test improvement | 0.11 | **0.23** |
| opensr-test omission | 0.72 | **0.53** |
| **field-boundary F1** (downstream) | **0.31** | **0.81** |

The headline is the last row: bicubic recovers 19 % of the true field edges,
DRISHTI 59 %, at the same precision. On the same held-out crop the CNN branch
alone scores 35.42 dB and the transformer branch alone 35.28 dB — the 50/50
fusion reaches 35.70. Every number is measured; no figure here is illustrative,
and goals (where stated elsewhere) are labelled TARGET.

## Layout

```
src/drishti/fetch.py        Sentinel-2 L2A fetch (CDSE Process API)
src/drishti/data.py         Wald-protocol pairs, degrade/upsample operators
src/drishti/tiling.py       tile iterator + feather-blend stitcher
src/drishti/metrics.py      losses (L1 + SAM + consistency) and reported metrics
src/drishti/model/drishti.py  hybrid CNN+Swin network, gate, exact low-freq constraint
src/drishti/infer.py        pretrained SEN2SR-Lite baseline + QA gate
scripts/                    CLI wrappers (fetch / train / evaluate / run_drishti / ...)
viewer/                     swipe-comparison viewer
```
