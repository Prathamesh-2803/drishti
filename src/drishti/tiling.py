"""Tile an image, run a per-tile operation, feather-blend the results back.

DRISHTI inference tiling (see CLAUDE.md):
    128x128 LR patches -> 512x512 SR patches, 16 px LR overlap, feather blend
    on stitch. A QA gate runs per tile BEFORE it is blended in; failed tiles
    are recorded and never stitched.

Everything here is scale-aware: a tile at LR position (y0, x0) lands at SR
position (y0 * scale, x0 * scale).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Tile:
    """One LR tile's placement. SR coords are LR coords * scale."""

    y0: int
    x0: int
    h: int  # LR tile height (may be < tile_size at bottom/right edges)
    w: int


def iter_tiles(height: int, width: int, tile: int, overlap: int) -> list[Tile]:
    """Cover (height, width) with `tile`-sized windows stepping by tile-overlap.

    Edge tiles are shifted back so they stay in bounds and keep full size
    (matches SEN2SR's fix_lastchunk behaviour).
    """
    if overlap >= tile:
        raise ValueError("overlap must be smaller than tile size")
    step = tile - overlap
    ys = list(range(0, max(height - overlap, 1), step))
    xs = list(range(0, max(width - overlap, 1), step))
    tiles: list[Tile] = []
    for y in ys:
        for x in xs:
            y0 = min(y, max(height - tile, 0))
            x0 = min(x, max(width - tile, 0))
            th = min(tile, height - y0)
            tw = min(tile, width - x0)
            tiles.append(Tile(y0=y0, x0=x0, h=th, w=tw))
    # De-duplicate (edge clamping can collide) while preserving order.
    seen: set[tuple[int, int]] = set()
    unique: list[Tile] = []
    for t in tiles:
        key = (t.y0, t.x0)
        if key not in seen:
            seen.add(key)
            unique.append(t)
    return unique


def feather_window(h: int, w: int, ramp: int) -> np.ndarray:
    """A 2-D weight window: 1.0 in the centre, linearly ramping to ~0 over
    `ramp` pixels at each edge. Used to blend overlapping SR tiles."""
    def axis(n: int) -> np.ndarray:
        r = min(ramp, n // 2)
        wgt = np.ones(n, dtype=np.float32)
        if r > 0:
            edge = (np.arange(r, dtype=np.float32) + 1.0) / (r + 1.0)
            wgt[:r] = edge
            wgt[-r:] = edge[::-1]
        return wgt

    return np.outer(axis(h), axis(w))


@dataclass
class Stitcher:
    """Weighted accumulator for feather-blending SR tiles into one raster."""

    bands: int
    sr_height: int
    sr_width: int
    scale: int
    ramp: int
    allow_gaps: bool = False
    gap_fill: float = float("nan")
    _acc: np.ndarray = field(init=False)
    _wsum: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self._acc = np.zeros(
            (self.bands, self.sr_height, self.sr_width), dtype=np.float64
        )
        self._wsum = np.zeros((self.sr_height, self.sr_width), dtype=np.float64)

    def add(self, tile: Tile, sr_patch: np.ndarray) -> None:
        """sr_patch: (bands, tile.h*scale, tile.w*scale)."""
        sy, sx = tile.y0 * self.scale, tile.x0 * self.scale
        ph, pw = sr_patch.shape[1], sr_patch.shape[2]
        win = feather_window(ph, pw, self.ramp * self.scale)
        self._acc[:, sy : sy + ph, sx : sx + pw] += sr_patch * win
        self._wsum[sy : sy + ph, sx : sx + pw] += win

    def result(self) -> np.ndarray:
        gap = self._wsum == 0
        if np.any(gap):
            if not self.allow_gaps:
                raise RuntimeError(
                    "stitch has uncovered pixels — some tiles were dropped by "
                    "the QA gate; re-run those tiles before stitching"
                )
            wsum = np.where(gap, 1.0, self._wsum)
            out = (self._acc / wsum).astype(np.float32)
            out[:, gap] = self.gap_fill
            return out
        return (self._acc / self._wsum).astype(np.float32)
