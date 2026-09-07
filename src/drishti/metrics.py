"""Losses and measurement for DRISHTI.

Two things live here and they are not the same thing:

  * losses -- what the optimiser minimises during training.
  * metrics -- what we report. Every reported number is measured on data the
    model did not train on. If a number has not been measured, it does not go
    in a table, a slide or a README.

Note on units, because it gets misquoted: PSNR is in decibels and is not
"accuracy". It is a log ratio of peak signal to mean squared error, so +3 dB
means the error was halved. SAM is in degrees. SSIM and the correctness terms
are unitless. The headline number for this project is the downstream task, not
PSNR.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .data import degrade, upsample


# --------------------------------------------------------------------------
# losses
# --------------------------------------------------------------------------
def sam_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Spectral Angle Mapper, mean over pixels, in radians.

    Penalises the model for rotating a pixel's spectral signature even when the
    brightness is right. This is the term that keeps NDVI and NDWI valid after
    4x, and it is the reason the output stays analysis-ready data rather than a
    prettier picture.
    """
    dot = (pred * target).sum(dim=1)
    npred = pred.pow(2).sum(dim=1).clamp_min(eps).sqrt()
    ntgt = target.pow(2).sum(dim=1).clamp_min(eps).sqrt()
    cos = (dot / (npred * ntgt)).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(cos).mean()


def consistency_loss(sr: torch.Tensor, lr: torch.Tensor, scale: int = 4, sigma: float | None = None):
    """Degrading the output must reproduce the input.

    With the hard constraint active this is already near zero; it is kept as a
    monitored quantity so a regression is visible immediately.
    """
    return F.l1_loss(degrade(sr, scale, sigma), lr)


def _sobel(x: torch.Tensor) -> torch.Tensor:
    """Per-channel gradient magnitude, reflect-padded."""
    kx = torch.tensor([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device, dtype=x.dtype)
    ky = kx.t()
    c = x.shape[1]
    kx = kx.view(1, 1, 3, 3).expand(c, 1, 3, 3)
    ky = ky.view(1, 1, 3, 3).expand(c, 1, 3, 3)
    xp = F.pad(x, (1, 1, 1, 1), mode="reflect")
    gx = F.conv2d(xp, kx, groups=c)
    gy = F.conv2d(xp, ky, groups=c)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def gradient_loss(sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
    """Edge-map L1. A cheap, download-free stand-in for a perceptual term: it
    rewards putting edges where the reference has edges and nowhere else, which
    is what a late sharpness push needs without a pretrained feature network."""
    return F.l1_loss(_sobel(sr), _sobel(hr))


class DrishtiLoss(torch.nn.Module):
    def __init__(self, w_l1: float = 1.0, w_sam: float = 0.15, w_cons: float = 0.5,
                 w_grad: float = 0.0, w_branch: float = 0.0, w_gate: float = 0.0,
                 scale: int = 4, sigma: float | None = None) -> None:
        super().__init__()
        self.w_l1, self.w_sam, self.w_cons, self.w_grad = w_l1, w_sam, w_cons, w_grad
        # w_branch: L1 on each branch's own reconstruction, so both the CNN and
        # the Swin path stay individually competent and the gate has a real
        # choice to make. w_gate: pull the gate's global mean toward balanced so
        # it cannot collapse onto one branch (it stays free to vary per pixel).
        self.w_branch, self.w_gate = w_branch, w_gate
        self.scale, self.sigma = scale, sigma

    def forward(self, sr: torch.Tensor, hr: torch.Tensor, lr: torch.Tensor, aux: dict | None = None):
        l1 = F.l1_loss(sr, hr)
        sam = sam_loss(sr, hr)
        cons = consistency_loss(sr, lr, self.scale, self.sigma)
        total = self.w_l1 * l1 + self.w_sam * sam + self.w_cons * cons
        parts = {"l1": l1.item(), "sam_rad": sam.item(), "cons": cons.item()}
        if self.w_grad:
            grad = gradient_loss(sr, hr)
            total = total + self.w_grad * grad
            parts["grad"] = grad.item()
        if aux and self.w_branch:
            br = F.l1_loss(aux["sr_cnn"], hr) + F.l1_loss(aux["sr_swin"], hr)
            total = total + self.w_branch * br
            parts["branch"] = br.item()
        if aux and self.w_gate:
            gm = aux["w"].mean()
            gbal = (gm - 0.5).pow(2)
            total = total + self.w_gate * gbal
            parts["gate_mean"] = gm.item()
        parts["total"] = total.item()
        return total, parts


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    mse = F.mse_loss(pred, target).item()
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10((data_range**2) / mse)


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return F.mse_loss(pred, target).sqrt().item()


def sam_degrees(pred: torch.Tensor, target: torch.Tensor) -> float:
    return math.degrees(sam_loss(pred, target).item())


def _gaussian_window(ws: int, sigma: float, ch: int, device, dtype):
    x = torch.arange(ws, dtype=dtype, device=device) - (ws - 1) / 2.0
    g = torch.exp(-(x**2) / (2 * sigma**2))
    g = (g / g.sum()).unsqueeze(0)
    win2d = (g.t() @ g).unsqueeze(0).unsqueeze(0)
    return win2d.expand(ch, 1, ws, ws).contiguous()


def ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0,
         ws: int = 11, sigma: float = 1.5) -> float:
    """Mean SSIM over channels, Gaussian-windowed, the standard formulation."""
    ch = pred.shape[1]
    win = _gaussian_window(ws, sigma, ch, pred.device, pred.dtype)
    pad = ws // 2
    mu1 = F.conv2d(pred, win, padding=pad, groups=ch)
    mu2 = F.conv2d(target, win, padding=pad, groups=ch)
    mu1s, mu2s, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = F.conv2d(pred * pred, win, padding=pad, groups=ch) - mu1s
    s2 = F.conv2d(target * target, win, padding=pad, groups=ch) - mu2s
    s12 = F.conv2d(pred * target, win, padding=pad, groups=ch) - mu12
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    m = ((2 * mu12 + c1) * (2 * s12 + c2)) / ((mu1s + mu2s + c1) * (s1 + s2 + c2))
    return m.mean().item()


def ndvi(x: torch.Tensor, red: int = 2, nir: int = 3, eps: float = 1e-6) -> torch.Tensor:
    """NDVI from the 4-band stack (B02 B03 B04 B08 -> red=2, nir=3)."""
    r, n = x[:, red : red + 1], x[:, nir : nir + 1]
    return (n - r) / (n + r + eps)


def evaluate_all(pred: torch.Tensor, target: torch.Tensor, lr: torch.Tensor,
                 scale: int = 4, sigma: float | None = None) -> dict:
    """Everything we report for one (prediction, truth) pair."""
    out = {
        "psnr_db": psnr(pred, target),
        "ssim": ssim(pred, target),
        "sam_deg": sam_degrees(pred, target),
        "rmse_reflectance": rmse(pred, target),
        # The shipped guarantee: block-average the SR back to the LR grid and it
        # must reproduce the input. With the hard constraint active this is ~1e-7.
        "consistency_blockavg_rmse": rmse(F.avg_pool2d(pred, scale), lr),
        # Secondary check against the (blurred) degradation the model trained on.
        "consistency_rmse": rmse(degrade(pred, scale, sigma), lr),
    }
    # Does a spectral index survive super-resolution? This is the claim that
    # separates an analysis-ready product from a nicer-looking picture.
    out["ndvi_mae"] = F.l1_loss(ndvi(pred), ndvi(target)).item()
    return out
