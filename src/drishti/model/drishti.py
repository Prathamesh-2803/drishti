"""The DRISHTI hybrid super-resolution network.

Two branches see the same shallow features and disagree usefully:

    CNN branch   residual channel-attention blocks (EDSR + RCAN lineage).
                 Local, translation-equivariant, data-efficient. Channel
                 attention matters more here than in ordinary photo SR because
                 the channels are spectral bands, not interchangeable colours --
                 near-infrared carries different evidence from red, and the
                 block learns which band to trust for a given structure.

    Swin branch  shifted-window self-attention (SwinIR lineage). Attention
                 weights are computed from content, so it can use a wider
                 neighbourhood than a 3x3 kernel ever sees. That is what
                 recovers repeated man-made structure -- building rows, field
                 grids, road networks.

A learned per-pixel gate mixes them:

    w = sigmoid(conv1x1([F_cnn, F_swin]))
    F = w * F_cnn + (1 - w) * F_swin

w is not an implementation detail. It is a map of where the transformer was
needed, it renders as an image, and it is one of the outputs.

Finally the low-frequency hard constraint. Whatever the network produces, the
result is forced to average back to the observed input:

    low_raw = avg_pool(SR_raw, scale)          # block mean on the LR grid
    SR      = SR_raw + nearest_up(LR - low_raw)

Because avg_pool(nearest_up(x)) == x identically, this guarantees
avg_pool(SR, scale) == LR to machine precision: block-averaging the 2.5 m
output back to 10 m reproduces the observed Sentinel-2 pixels exactly. Large-
scale structure and radiometry come from the real observation and only fine
detail comes from the model. Hallucination is bounded by construction rather
than discouraged by a loss term.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# CNN branch
# --------------------------------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, ch: int, reduction: int = 8) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(ch, max(ch // reduction, 4), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(ch // reduction, 4), ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x.mean(dim=(2, 3), keepdim=True))


class RCAB(nn.Module):
    """Residual channel-attention block."""

    def __init__(self, ch: int, res_scale: float = 0.1) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            ChannelAttention(ch),
        )
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.res_scale * self.body(x)


class CNNBranch(nn.Module):
    def __init__(self, ch: int = 64, n_blocks: int = 8) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[RCAB(ch) for _ in range(n_blocks)])
        self.tail = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.tail(self.blocks(x))


# --------------------------------------------------------------------------
# Swin branch
# --------------------------------------------------------------------------
def window_partition(x: torch.Tensor, ws: int) -> torch.Tensor:
    """(B, H, W, C) -> (B*nW, ws*ws, C)"""
    b, h, w, c = x.shape
    x = x.view(b, h // ws, ws, w // ws, ws, c)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, c)


def window_reverse(win: torch.Tensor, ws: int, h: int, w: int) -> torch.Tensor:
    """(B*nW, ws*ws, C) -> (B, H, W, C)"""
    b = win.shape[0] // ((h // ws) * (w // ws))
    x = win.view(b, h // ws, w // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(b, h, w, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim: int, ws: int, heads: int) -> None:
        super().__init__()
        self.dim, self.ws, self.heads = dim, ws, heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        # learned relative position bias, the piece that makes windows work
        self.bias = nn.Parameter(torch.zeros((2 * ws - 1) ** 2, heads))
        coords = torch.stack(
            torch.meshgrid(torch.arange(ws), torch.arange(ws), indexing="ij")
        ).flatten(1)
        rel = coords[:, :, None] - coords[:, None, :]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += ws - 1
        rel[:, :, 1] += ws - 1
        rel[:, :, 0] *= 2 * ws - 1
        self.register_buffer("rel_index", rel.sum(-1), persistent=False)
        nn.init.trunc_normal_(self.bias, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.heads, c // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.bias[self.rel_index.view(-1)].view(n, n, -1).permute(2, 0, 1)
        attn = attn + bias.unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.heads, n, n)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        return self.proj(out)


class SwinBlock(nn.Module):
    def __init__(self, dim: int, ws: int = 8, heads: int = 4, shift: int = 0, mlp: float = 2.0) -> None:
        super().__init__()
        self.ws, self.shift = ws, shift
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, ws, heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def _mask(self, h: int, w: int, device) -> torch.Tensor | None:
        if self.shift == 0:
            return None
        img = torch.zeros((1, h, w, 1), device=device)
        cnt = 0
        for hs in (slice(0, -self.ws), slice(-self.ws, -self.shift), slice(-self.shift, None)):
            for wsl in (slice(0, -self.ws), slice(-self.ws, -self.shift), slice(-self.shift, None)):
                img[:, hs, wsl, :] = cnt
                cnt += 1
        win = window_partition(img, self.ws).squeeze(-1)
        att = win.unsqueeze(1) - win.unsqueeze(2)
        return att.masked_fill(att != 0, -100.0).masked_fill(att == 0, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, W, C)"""
        b, h, w, c = x.shape
        skip = x
        x = self.norm1(x)
        if self.shift:
            x = torch.roll(x, (-self.shift, -self.shift), dims=(1, 2))
        win = window_partition(x, self.ws)
        win = self.attn(win, self._mask(h, w, x.device))
        x = window_reverse(win, self.ws, h, w)
        if self.shift:
            x = torch.roll(x, (self.shift, self.shift), dims=(1, 2))
        x = skip + x
        return x + self.mlp(self.norm2(x))


class SwinBranch(nn.Module):
    def __init__(self, ch: int = 64, depth: int = 4, ws: int = 8, heads: int = 4) -> None:
        super().__init__()
        self.ws = ws
        self.blocks = nn.ModuleList(
            [SwinBlock(ch, ws, heads, shift=0 if i % 2 == 0 else ws // 2) for i in range(depth)]
        )
        self.tail = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W). Pads to a whole number of windows, then crops back."""
        b, c, h, w = x.shape
        ph = (self.ws - h % self.ws) % self.ws
        pw = (self.ws - w % self.ws) % self.ws
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        y = x.permute(0, 2, 3, 1).contiguous()
        for blk in self.blocks:
            y = blk(y)
        y = y.permute(0, 3, 1, 2).contiguous()
        if ph or pw:
            y = y[:, :, :h, :w]
            x = x[:, :, :h, :w]
        return x + self.tail(y)


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------
class Drishti(nn.Module):
    def __init__(
        self,
        bands: int = 4,
        ch: int = 64,
        cnn_blocks: int = 8,
        swin_depth: int = 4,
        window: int = 8,
        heads: int = 4,
        scale: int = 4,
        sigma: float | None = None,
        dropout: float = 0.1,
        hard_constraint: bool = True,
    ) -> None:
        super().__init__()
        self.scale, self.sigma = scale, sigma
        self.hard_constraint = hard_constraint
        self.head = nn.Conv2d(bands, ch, 3, padding=1)
        self.cnn = CNNBranch(ch, cnn_blocks)
        self.swin = SwinBranch(ch, swin_depth, window, heads)
        self.gate = nn.Conv2d(ch * 2, ch, 1)
        # dropout is what makes the confidence map possible: kept active at
        # inference for N stochastic passes, the spread is the uncertainty.
        self.drop = nn.Dropout2d(dropout)
        self.fuse = nn.Conv2d(ch, ch, 3, padding=1)
        up = []
        for _ in range(int(torch.log2(torch.tensor(float(scale))).item())):
            up += [nn.Conv2d(ch, ch * 4, 3, padding=1), nn.PixelShuffle(2), nn.ReLU(inplace=True)]
        self.up = nn.Sequential(*up)
        self.tail = nn.Conv2d(ch, bands, 3, padding=1)

    def _constrain(self, raw: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
        """Exact low-frequency projection: avg_pool2d(nearest_up(x)) == x, so
        avg_pool2d(sr, scale) == lr to machine precision regardless of what the
        network produced (CLAUDE.md rule 4)."""
        if not self.hard_constraint:
            return raw
        low_raw = F.avg_pool2d(raw, self.scale)
        corr = F.interpolate(lr - low_raw, scale_factor=self.scale, mode="nearest")
        return raw + corr

    def _decode(self, feat: torch.Tensor, f0: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
        return self._constrain(self.tail(self.up(self.fuse(feat) + f0)), lr)

    def reset_gate(self) -> None:
        """Zero the gate so it starts balanced (w == 0.5 everywhere). Used before
        a fine-tune when the loaded gate has collapsed to one branch."""
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, lr: torch.Tensor, return_gate: bool = False, return_branches: bool = False):
        f0 = self.head(lr)
        f_cnn = self.cnn(f0)
        f_swin = self.swin(f0)
        w = torch.sigmoid(self.gate(torch.cat([f_cnn, f_swin], dim=1)))
        f = w * f_cnn + (1.0 - w) * f_swin
        f = self.fuse(self.drop(f)) + f0
        sr = self._constrain(self.tail(self.up(f)), lr)
        gate_map = w.mean(dim=1, keepdim=True)

        out = [sr]
        if return_gate:
            out.append(gate_map)
        if return_branches:
            # each branch decoded through the shared head, so an auxiliary loss
            # can hold both individually competent and stop the gate collapsing
            out.append({"w": gate_map,
                        "sr_cnn": self._decode(f_cnn, f0, lr),
                        "sr_swin": self._decode(f_swin, f0, lr)})
        return out[0] if len(out) == 1 else tuple(out)

    @torch.no_grad()
    def predict_with_uncertainty(self, lr: torch.Tensor, passes: int = 8):
        """Mean SR and per-pixel standard deviation over N dropout passes.

        Dropout stays on deliberately. Where the passes agree the model is
        confident; where they disagree it was guessing, and that is exactly what
        the confidence raster reports.
        """
        was_training = self.training
        self.eval()
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()
        outs = torch.stack([self(lr) for _ in range(passes)], dim=0)
        self.train(was_training)
        return outs.mean(0), outs.std(0).mean(dim=1, keepdim=True)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
