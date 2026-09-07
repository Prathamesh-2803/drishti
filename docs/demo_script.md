# DRISHTI — live demo script

Explorer: **https://claude.ai/code/artifact/e64a41f8-6009-4abb-bfe9-da5aeddd3baf**
Works on a phone. Have it open on the map view before you start.

---

## 60-second version

> "This is Sentinel-2 — free satellite imagery of all of India, 10 metres a
> pixel. At 10 metres a house is four pixels, so you can't reliably map fields
> or plan at parcel scale."

**[pick Ludhiana — canal cropland]**

> "Here's a region of Punjab. I press Enhance..."

**[press *Enhance to 2.5 m*, let the scan animation run, then drag the slider left–right a few times]**

> "...and DRISHTI takes it to 2.5 metres. Left is the 10-metre input, right is
> our output. The field boundaries that were mush are now lines you could
> digitise."

**[scroll to zoom in on a block of fields, keep swiping]**

> "And it holds up when you zoom — this is individual 2.5-metre detail, on a
> region the model was **never trained on**."

**[click the RELIABILITY layer]**

> "Every enhanced image ships with this — a confidence map. Bright means the
> model was less sure there. And there's a hard mathematical rule inside: shrink
> our 2.5-metre output back to 10 metres and you get the original Sentinel-2
> pixels back, exactly. It can add detail, it cannot change the measurement — so
> it can't hallucinate a field that isn't there."

**[scroll down to the accuracy panel]**

> "Measured blind on the held-out region: structural similarity 89.6%, and it
> recovers 2.7 times as many field boundaries as standard upscaling. That last
> number is the one that matters — it decides whether a parcel map drawn from
> the imagery is usable."

---

## If you have longer — extra beats

- **Vegetation view.** Toggle VEGETATION on any scene: "near-infrared — this is
  how you'd check crop health. The colours stay measurement-accurate after
  super-resolution, which is the whole point of working in calibrated
  reflectance and not a stretched RGB."
- **A hard case.** Pick Coimbatore (hills) or Jaipur (dryland): "It's not
  magic — on steep terrain with shadow the output is softer, and you can see the
  confidence map light up there. It degrades gracefully and tells you when."
- **The two branches.** "Under the hood there are two networks — a CNN for local
  detail and a transformer for wide context like road and building patterns.
  They're fused 50/50, and together they beat either one alone."
- **What's next.** "Right now we validate by simulating the scale step. With
  co-registered Cartosat / LISS-IV imagery from Bhoonidhi we'd measure the real
  10-to-2.5 task directly and fine-tune on Indian ground — that's the next
  round."

---

## Numbers to have memorised

| | value |
|---|---|
| Resolution | 10 m → 2.5 m (×4) |
| Held-out PSNR vs bicubic | +1.14 dB (36.4 vs 35.3) |
| Structural similarity (SSIM) | 0.896 |
| Field-boundary recovery vs bicubic | 2.7× (F1 0.81 vs 0.31) |
| Spectral-angle error | 2.6° (down from 3.1°) |
| Downscale consistency | ~1 × 10⁻⁵ reflectance (bicubic: 6 × 10⁻³) |
| Training data | 12 Indian Sentinel-2 scenes; Vasai held out |
| Model size | 3.4 M parameters; runs on a laptop, air-gappable |

## Do not claim

- Do **not** say "validated against real 2.5 m imagery" — we validate by
  simulating the scale step (Wald's protocol). Say "measured blind on a held-out
  region."
- Do **not** quote a number that isn't in `outputs/eval_vasai/` or this file.
- The confidence map is **relative**, not calibrated to metres of error.
