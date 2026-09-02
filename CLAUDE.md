# DRISHTI — Sentinel-2 Super-Resolution (SIH 2026, PS 26142)

Turn free 10 m Sentinel-2 imagery into 2.5 m analysis-ready products, with a
per-pixel confidence map. Team THE OUTLIERS. Org: NTRO. Theme: Space Technology.
PS requires: 10 m in, <4 m out, geospatial + spectral consistency preserved,
uncertainty explicitly managed, validated against high-resolution references.

## Non-negotiable rules
1. NEVER report a metric that has not been measured. No illustrative numbers in
   code, docs, plots or slides. If a number is a goal, label it TARGET.
2. Work in surface reflectance (L2A), never 8-bit stretched RGB.
3. Scale is 4x: 10 m -> 2.5 m. Do not introduce other scale factors.
4. Low-frequency hard constraint is mandatory:
   SR_final = upsample(LR) + highpass(SR_raw), so degrade(SR) reproduces LR.
5. The confidence raster ships with every image. It is a product, not a debug artifact.

## Architecture (decided)
Hybrid two-branch with gated fusion:
- shallow conv: 10 bands -> features
- CNN branch: EDSR + RCAN (residual-in-residual, channel attention)
- Swin branch: SwinIR RSTBs (window + shifted-window attention)
- gate: w = sigmoid(conv1x1([F_cnn, F_swin])); F = w*F_cnn + (1-w)*F_swin + shallow
- pixel shuffle x4 -> conv -> 10 bands -> low-frequency constraint
- MC-dropout, N passes -> per-pixel variance -> confidence map
The gate map is a deliverable: render it to show where the transformer was needed.

## Losses
L1 in reflectance + Spectral Angle Mapper + downsample-consistency, plus a small
adversarial/perceptual term added late with low weight.

## Data
LR: Sentinel-2 L2A via Copernicus Data Space (OData / STAC / Process API).
Pretrain pairs: SEN2NAIP v2, SEN2VENuS.
India fine-tune: Bhoonidhi (NRSC) LISS-IV 5.8 m, Cartosat-1 PAN 2.5 m.
Co-register every real pair (phase correlation / AROSICS) before training on it.

## Tiling
Train: 64x64 LR patches, batch 16, AMP on.
Infer: 128x128 LR -> 512x512, 16 px overlap, feather blend on stitch.
A QA gate runs per tile BEFORE stitching (consistency, SAM, confidence).
Failed tiles are re-run, never stitched.

## Metrics
PSNR (dB), SSIM, LPIPS, SAM (degrees), reflectance L1, opensr-test
hallucination/omission/improvement, plus one downstream task
(field-boundary IoU) measured before vs after.
PSNR is NOT accuracy. The headline number is the downstream task.

## Build order (do not reorder)
1. Register: Copernicus Data Space, Bhoonidhi. Email Planet Education & Research.
2. Download one Sentinel-2 patch and one LISS-IV patch of the same place; open both in QGIS.
3. Run an open pretrained SR model on our own AOI, end to end.
4. Build the swipe-comparison viewer. It is the debugging tool, not just a demo.
5. Only then fine-tune on Indian pairs, add the gate, add the confidence map.

## Stack
Python, PyTorch, rasterio/GDAL, Cloud-Optimised GeoTIFF, FastAPI, MapLibre viewer,
Docker (must be able to run air-gapped).

## Status
Nothing trained yet. The deck and technical-approach figure exist; the imagery in
them is SIMULATED and must be replaced with real output once step 3 produces it.
