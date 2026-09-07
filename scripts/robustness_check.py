"""Robustness spot-check: run DRISHTI on deliberately hard 640 m tiles and
measure where it holds and where it degrades. Wald protocol per tile."""
import sys
sys.path.insert(0, "src")
import numpy as np, rasterio, torch
from rasterio.warp import transform as wtf
from drishti.data import RGBN, degrade, upsample
from drishti.metrics import psnr, ssim, sam_degrees, ndvi
from drishti.model.drishti import Drishti
from PIL import Image, ImageDraw, ImageFont
from matplotlib import cm

ROOT = "/Users/avadhmehta/Projects/drishti"
ck = torch.load(f"{ROOT}/checkpoints/drishti-ft/best.pt", map_location="cpu", weights_only=False)
a = ck["args"]
m = Drishti(bands=4, ch=a["ch"], cnn_blocks=a["cnn_blocks"], swin_depth=a["swin_depth"],
            window=a["window"], scale=4, sigma=0.0, dropout=a["dropout"])
m.load_state_dict(ck["model"]); m.eval()

# (scene, lon, lat, label) — spots chosen for a specific hard condition
CASES = [
    ("vasai", 72.83, 19.36, "coastal tidal flats + salt pans (held-out scene)"),
    ("bhimavaram", 81.55, 16.56, "open water + aquaculture ponds"),
    ("coimbatore", 76.97, 11.06, "Western Ghats ridge (steep relief + shadow)"),
    ("jaipur", 75.80, 26.96, "bare rock / very low contrast dryland"),
    ("mumbai", 72.87, 19.05, "high-rise urban core (tall buildings, hard shadows)"),
]

rows = []
tiles_img = []
for scene, lon, lat, label in CASES:
    p = f"{ROOT}/data/raw/{scene}_s2l2a.tif"
    with rasterio.open(p) as d:
        xs, ys = wtf("EPSG:4326", d.crs, [lon], [lat])
        col, row = (~d.transform) * (xs[0], ys[0])
        n = 64  # 640 m at 10 m
        col = int(min(max(col, n), d.width - n))
        row = int(min(max(row, n), d.height - n))
        desc = list(d.descriptions)
        idx = [desc.index(b) + 1 for b in RGBN]
        win = rasterio.windows.Window(col - n, row - n, 2 * n, 2 * n)
        arr = d.read(idx, window=win).astype(np.float32)
        sf = float(d.tags().get("reflectance_scale_factor", 1.0) or 1.0)
    arr = np.clip(arr * sf, 0, 1.5)
    hr = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0)
    lr = degrade(hr, 4, 0.0)
    with torch.no_grad():
        mean, std = m.predict_with_uncertainty(lr, passes=16)
        sr = m(lr)
    bic = upsample(lr, 4)
    r = {
        "case": label, "scene": scene,
        "psnr_bic": psnr(bic, hr), "psnr_sr": psnr(sr, hr),
        "ssim_bic": ssim(bic, hr), "ssim_sr": ssim(sr, hr),
        "sam_bic": sam_degrees(bic, hr), "sam_sr": sam_degrees(sr, hr),
        "ndvi_bic": torch.nn.functional.l1_loss(ndvi(bic), ndvi(hr)).item(),
        "ndvi_sr": torch.nn.functional.l1_loss(ndvi(sr), ndvi(hr)).item(),
        "cons": float((torch.nn.functional.avg_pool2d(sr, 4) - lr).abs().mean()),
        "conf_mean": float(std.mean()), "conf_p95": float(np.percentile(std.numpy(), 95)),
    }
    rows.append(r)
    print(f"\n{label}  ({scene})")
    print(f"  PSNR  bicubic {r['psnr_bic']:.2f}  DRISHTI {r['psnr_sr']:.2f}  ({r['psnr_sr']-r['psnr_bic']:+.2f} dB)")
    print(f"  SSIM  {r['ssim_bic']:.3f} -> {r['ssim_sr']:.3f}   SAM {r['sam_bic']:.2f} -> {r['sam_sr']:.2f} deg")
    print(f"  NDVI L1 {r['ndvi_bic']:.4f} -> {r['ndvi_sr']:.4f}   downscale-consistency {r['cons']:.1e}")
    print(f"  confidence: mean {r['conf_mean']:.5f}  p95 {r['conf_p95']:.5f}  (scene-typical p95 ~5e-4)")

    lo = [np.percentile(arr[i], 2) for i in range(4)]; hi = [np.percentile(arr[i], 98) for i in range(4)]
    def rgb(c):
        return (np.stack([np.clip((c[i]-lo[i])/(hi[i]-lo[i]), 0, 1) for i in [2, 1, 0]], -1)*255).astype("uint8")
    lrn = lr[0].numpy()
    up = np.repeat(np.repeat(lrn, 4, 1), 4, 2)  # the actual 40 m model input, x4 nearest
    cf = std[0, 0].numpy()
    cfi = (cm.inferno(np.clip(cf/max(np.percentile(cf, 99), 1e-6), 0, 1))[..., :3]*255).astype("uint8")
    tiles_img.append((label, [rgb(up), rgb(bic[0].numpy()), rgb(sr[0].numpy()), rgb(hr[0].numpy()), cfi], r))

# montage
C = 250
cv = Image.new("RGB", (5*C+30, len(tiles_img)*(C+34)+30), "#0a0e13")
d = ImageDraw.Draw(cv)
f = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 12)
fb = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 13)
labs = ["40 m input (x4)", "bicubic", "DRISHTI", "truth 10 m", "confidence"]
for r_i, (label, imgs, rr) in enumerate(tiles_img):
    y = 30 + r_i*(C+34)
    d.text((14, y-20), f"{label}   |   PSNR {rr['psnr_bic']:.1f} -> {rr['psnr_sr']:.1f} dB   SSIM {rr['ssim_bic']:.2f} -> {rr['ssim_sr']:.2f}   consistency {rr['cons']:.0e}",
           fill="#cdd6df", font=fb)
    for c_i, im in enumerate(imgs):
        x = 14 + c_i*(C+2)
        cv.paste(Image.fromarray(im).resize((C, C), Image.NEAREST), (x, y))
        if r_i == 0:
            d.text((x, y-34), labs[c_i], fill="#8b98a7", font=f)
cv.save("outputs/robustness.png")
print("\nsaved robustness.png", cv.size)
