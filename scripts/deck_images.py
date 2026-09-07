"""Real slide-2 imagery for the DRISHTI deck, from the fixed-model Vasai output."""
import numpy as np, rasterio
from PIL import Image, ImageDraw, ImageFont
from matplotlib import cm

ROOT = "/Users/avadhmehta/Projects/drishti"
OUT = "outputs/deck"

def rd(p, b=None):
    with rasterio.open(p) as d:
        a = (d.read(b) if b else d.read()).astype(np.float32)
        sf = float(d.tags().get("reflectance_scale_factor", 1.0) or 1.0)
    return a * sf

sr = rd(f"{ROOT}/outputs/sr/vasai.tif")                       # (4, H, W) 2.5 m
raw = rd(f"{ROOT}/data/raw/vasai_s2l2a.tif", [1, 2, 3, 7])     # B02 B03 B04 B08 10 m
lr_up = np.repeat(np.repeat(raw, 4, 1), 4, 2)[:, : sr.shape[1], : sr.shape[2]]
conf = rd(f"{ROOT}/outputs/sr/vasai_confidence.tif")[0]

lo = [np.percentile(sr[i], 2) for i in range(4)]
hi = [np.percentile(sr[i], 98) for i in range(4)]
def rgb(c, sl):
    return (np.stack([np.clip((c[i][sl] - lo[i]) / (hi[i] - lo[i]), 0, 1) for i in [2, 1, 0]], -1) * 255).astype("uint8")

H, W = sr.shape[1], sr.shape[2]
try:
    fB = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 26)
    fS = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 22)
    fT = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 21)
except Exception:
    fB = fS = fT = ImageFont.load_default()

# ---------- SWIPE  (2.56 : 1, id=104) ----------
tw, th = 1660, 648
# a wide crop: coast/salt-pans left, built-up Vasai right
ch = int(th / tw * (W))          # crop height in SR px to match aspect if we take full width
cw = W
y0 = int(H * 0.34)
ch = int(cw * th / tw)
sl = (slice(y0, y0 + ch), slice(0, cw))
left = Image.fromarray(rgb(lr_up, sl)).resize((tw, th), Image.NEAREST)
right = Image.fromarray(rgb(sr, sl)).resize((tw, th), Image.LANCZOS)
canvas = Image.new("RGB", (tw, th))
canvas.paste(left, (0, 0))
canvas.paste(right.crop((tw // 2, 0, tw, th)), (tw // 2, 0))
d = ImageDraw.Draw(canvas)
d.line((tw // 2, 0, tw // 2, th), fill=(242, 162, 78), width=3)
# handle
d.ellipse((tw // 2 - 20, th // 2 - 20, tw // 2 + 20, th // 2 + 20), outline=(242, 162, 78), width=3, fill=(10, 12, 16))
d.text((tw // 2 - 15, th // 2 - 13), "‹ ›", font=fT, fill=(242, 162, 78))
def tag(x, y, s, anchor_r=False):
    w = d.textlength(s, font=fT)
    x0 = x - w - 16 if anchor_r else x
    d.rectangle((x0 - 8, y - 6, x0 + w + 10, y + 28), fill=(0, 0, 0))
    d.text((x0 + 2, y - 2), s, font=fT, fill=(255, 255, 255))
tag(14, 14, "SENTINEL-2  ·  10 m")
tag(tw - 14, 14, "DRISHTI  ·  2.5 m", anchor_r=True)
# measured ribbon
rb = 44
d.rectangle((0, th - rb, tw, th), fill=(12, 15, 20))
d.text((16, th - rb + 10), "MEASURED, blind on held-out Vasai:", font=fT, fill=(240, 162, 83))
d.text((16 + d.textlength("MEASURED, blind on held-out Vasai:", font=fT) + 22, th - rb + 11),
       "+1.1 dB vs bicubic   ·   SSIM 0.90   ·   2.7× more field boundaries   ·   shrinks back to the input exactly",
       font=fS, fill=(210, 216, 222))
canvas.save(f"{OUT}/deck_slide2_swipe.png")
print("swipe", canvas.size)

# ---------- STRIP  (4.8 : 1, id=105) : 3 tight panels ----------
sw, sh = 1660, 346
cs = 900
cy, cx = int(H * 0.42), int(W * 0.36)
c = (slice(cy, cy + cs), slice(cx, cx + cs))
panels = [
    ("10 m input", Image.fromarray(rgb(lr_up, c)).resize((sh, sh), Image.NEAREST)),
    ("DRISHTI  2.5 m", Image.fromarray(rgb(sr, c)).resize((sh, sh), Image.LANCZOS)),
]
cfc = conf[c]
cfi = (cm.inferno(np.clip(cfc / max(np.percentile(cfc, 99), 1e-6), 0, 1))[..., :3] * 255).astype("uint8")
panels.append(("confidence  (amber = a guess)", Image.fromarray(cfi).resize((sh, sh), Image.LANCZOS)))
strip = Image.new("RGB", (sw, sh), (12, 15, 20))
gap = (sw - 3 * sh) // 4
d = ImageDraw.Draw(strip)
for i, (lab, im) in enumerate(panels):
    x = gap + i * (sh + gap)
    strip.paste(im, (x, 0))
    tw2 = d.textlength(lab, font=fS)
    d.rectangle((x, sh - 32, x + tw2 + 14, sh), fill=(0, 0, 0))
    d.text((x + 6, sh - 30, ), lab, font=fS, fill=(255, 255, 255))
strip.save(f"{OUT}/deck_slide2_strip.png")
print("strip", strip.size)
