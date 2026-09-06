"""Paired training data for DRISHTI, built under Wald's protocol.

We have no co-registered 2.5 m reference for Indian ground (Bhoonidhi approval
is not instant), so we cannot supervise 10 m -> 2.5 m directly. Wald's protocol
is the standard remote-sensing answer: train the operator one scale step down,
where perfect ground truth exists by construction, then apply the same operator
at the target scale.

    HR target = the real Sentinel-2 scene at 10 m
    LR input  = that same scene degraded to 40 m with a sensor-like blur

The model therefore learns 40 m -> 10 m with exact supervision. At inference we
apply it to the untouched 10 m scene and obtain 2.5 m. The assumption being made
is scale invariance of the degradation operator, and it is stated openly rather
than hidden.

Reflectance is kept in physical units throughout (roughly 0..1). Nothing is
stretched to 8-bit; the radiometry has to survive the pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Band order written by drishti.fetch. The 4-band model uses blue/green/red/NIR:
# NDVI needs only red and NIR, so this subset is enough to show that a spectral
# index still works after super-resolution.
ALL_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
RGBN = ["B02", "B03", "B04", "B08"]


def band_indices(bands: list[str], available: list[str] | None = None) -> list[int]:
    """0-based indices of `bands` within the fetched band order."""
    avail = available or ALL_BANDS
    missing = [b for b in bands if b not in avail]
    if missing:
        raise ValueError(f"bands not present in scene: {missing}")
    return [avail.index(b) for b in bands]


# --------------------------------------------------------------------------
# degradation operator
# --------------------------------------------------------------------------
def gaussian_kernel1d(sigma: float, radius: int | None = None) -> torch.Tensor:
    """Normalised 1-D Gaussian, used separably."""
    if radius is None:
        radius = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-(x**2) / (2.0 * sigma * sigma))
    return k / k.sum()


def blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur on (B, C, H, W), reflect-padded."""
    if sigma <= 0:
        return x
    k = gaussian_kernel1d(sigma).to(x.device, x.dtype)
    r = (k.numel() - 1) // 2
    c = x.shape[1]
    kx = k.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    ky = k.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    x = F.pad(x, (r, r, 0, 0), mode="reflect")
    x = F.conv2d(x, kx, groups=c)
    x = F.pad(x, (0, 0, r, r), mode="reflect")
    x = F.conv2d(x, ky, groups=c)
    return x


def degrade(x: torch.Tensor, scale: int = 4, sigma: float | None = None) -> torch.Tensor:
    """HR -> LR: sensor-like blur, then area average.

    sigma defaults to 0.4 * scale, which puts the MTF at the new Nyquist in the
    range Sentinel-2's own optics sit in. A plain bicubic downsample would teach
    the model to invert bicubic rather than to invert a satellite, so the blur is
    not optional.
    """
    if sigma is None:
        sigma = 0.4 * scale
    return F.avg_pool2d(blur(x, sigma), kernel_size=scale, stride=scale)


def upsample(x: torch.Tensor, scale: int = 4) -> torch.Tensor:
    """LR -> HR grid, bilinear. Used for the low-frequency path only."""
    return F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False)


# --------------------------------------------------------------------------
# scenes
# --------------------------------------------------------------------------
@dataclass
class Scene:
    name: str
    array: np.ndarray  # (C, H, W) float32 reflectance
    path: str


def load_scene(path: str | Path, bands: list[str] = RGBN) -> Scene:
    """Read a fetched GeoTIFF and return the requested bands as float32."""
    import rasterio

    path = Path(path)
    with rasterio.open(path) as ds:
        desc = [d for d in (ds.descriptions or []) if d]
        available = desc if len(desc) == ds.count else ALL_BANDS[: ds.count]
        idx = band_indices(bands, available)
        arr = ds.read([i + 1 for i in idx]).astype(np.float32)
    # Reflectance guard: if the file is in scaled integers, bring it to 0..1.
    if np.nanmax(arr) > 10.0:
        arr = arr / 10000.0
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return Scene(name=path.stem.replace("_s2l2a", ""), array=np.clip(arr, 0.0, 1.5), path=str(path))


class WaldPatches(torch.utils.data.Dataset):
    """Random HR crops from a set of scenes; LR is made on the fly.

    Cropping in HR and degrading afterwards (rather than pre-degrading the whole
    scene) keeps the LR/HR pair perfectly registered, which is the one thing that
    would otherwise poison the training signal.
    """

    def __init__(
        self,
        scenes: list[Scene],
        hr_patch: int = 256,
        scale: int = 4,
        length: int = 4000,
        sigma: float | None = None,
        augment: bool = True,
        seed: int = 0,
        min_valid: float = 0.05,
    ) -> None:
        if hr_patch % scale:
            raise ValueError("hr_patch must be divisible by scale")
        self.scenes = scenes
        self.hr = hr_patch
        self.scale = scale
        self.length = length
        self.sigma = sigma
        self.augment = augment
        self.min_valid = min_valid
        self.rng = np.random.default_rng(seed)
        for s in scenes:
            if s.array.shape[1] < hr_patch or s.array.shape[2] < hr_patch:
                raise ValueError(f"scene {s.name} smaller than hr_patch {hr_patch}")

    def __len__(self) -> int:
        return self.length

    def _crop(self) -> np.ndarray:
        """One HR crop that is not almost entirely no-data."""
        for _ in range(20):
            s = self.scenes[self.rng.integers(len(self.scenes))]
            _, H, W = s.array.shape
            y = int(self.rng.integers(0, H - self.hr + 1))
            x = int(self.rng.integers(0, W - self.hr + 1))
            patch = s.array[:, y : y + self.hr, x : x + self.hr]
            if float(patch.mean()) > self.min_valid:
                return patch.copy()
        return patch.copy()

    def __getitem__(self, _: int) -> tuple[torch.Tensor, torch.Tensor]:
        patch = self._crop()
        if self.augment:
            k = int(self.rng.integers(4))
            if k:
                patch = np.rot90(patch, k, axes=(1, 2)).copy()
            if self.rng.random() < 0.5:
                patch = patch[:, :, ::-1].copy()
            if self.rng.random() < 0.5:
                patch = patch[:, ::-1, :].copy()
        hr = torch.from_numpy(patch)
        lr = degrade(hr.unsqueeze(0), self.scale, self.sigma).squeeze(0)
        return lr, hr


def scene_eval_pair(
    scene: Scene, scale: int = 4, sigma: float | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Whole-scene (LR, HR) pair for held-out evaluation.

    The HR side is cropped so its size is an exact multiple of `scale`, so that
    degrade/upsample round-trip without an off-by-one.
    """
    _, H, W = scene.array.shape
    H -= H % scale
    W -= W % scale
    hr = torch.from_numpy(scene.array[:, :H, :W]).unsqueeze(0)
    lr = degrade(hr, scale, sigma)
    return lr, hr


# --------------------------------------------------------------------------
# spatial splitting
# --------------------------------------------------------------------------
def crop_scene(scene: Scene, y0: int, y1: int, suffix: str = "") -> Scene:
    """A row-range of a scene, as a Scene in its own right."""
    return Scene(name=scene.name + suffix,
                 array=np.ascontiguousarray(scene.array[:, y0:y1, :]),
                 path=scene.path)


def split_scene(scene: Scene, holdout_frac: float = 0.25, gap: int = 64
                ) -> tuple[Scene, Scene]:
    """Split one scene into a training region and a held-out region.

    Used when there is only one scene available, or when cross-region scenes
    cannot be fetched. The `gap` rows between the two are discarded and belong
    to neither: satellite pixels are spatially autocorrelated, so a held-out
    strip that touches the training region is not really held out, and a model
    can score well on it by memorising the boundary.

    Train takes the north, holdout takes the south, so the two never overlap.
    """
    _, H, _ = scene.array.shape
    n_hold = int(round(H * holdout_frac))
    if n_hold <= gap:
        raise ValueError("holdout region too small for the requested gap")
    train_end = H - n_hold
    hold_start = train_end + gap
    if hold_start >= H:
        raise ValueError("gap leaves no held-out rows")
    return (crop_scene(scene, 0, train_end, "_train"),
            crop_scene(scene, hold_start, H, "_holdout"))
