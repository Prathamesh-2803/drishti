#!/usr/bin/env python3
"""Assert the low-frequency hard constraint is exact (CLAUDE.md rule 4).

Block-averaging the SR output back to the LR grid must reproduce the input to
machine precision, for any input and any (untrained) weights.

    python scripts/check_constraint.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drishti.data import RGBN  # noqa: E402
from drishti.model.drishti import Drishti  # noqa: E402


def main() -> int:
    torch.manual_seed(0)
    scale = 4
    model = Drishti(bands=len(RGBN), ch=16, cnn_blocks=2, swin_depth=2, scale=scale)
    model.eval()

    worst = 0.0
    for h, w in [(16, 16), (32, 24), (48, 48), (64, 64), (33, 41)]:
        lr = torch.rand(2, len(RGBN), h, w)
        with torch.no_grad():
            sr = model(lr)
        assert sr.shape[-2:] == (h * scale, w * scale), sr.shape
        err = (F.avg_pool2d(sr, scale) - lr).abs().max().item()
        worst = max(worst, err)
        print(f"  lr {h:>3}x{w:<3}  ->  sr {tuple(sr.shape[-2:])}  max|avg_pool(sr)-lr| = {err:.2e}")

    tol = 1e-4
    print(f"\nworst: {worst:.2e}  (tol {tol:.0e})")
    if worst > tol:
        print("FAIL: hard constraint is not exact")
        return 1
    print("OK: avg_pool2d(SR, scale) reproduces LR exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
