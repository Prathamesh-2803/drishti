#!/usr/bin/env python3
"""Inline frontend/index.html + data.js + assets into one standalone HTML file
for publishing as an Artifact. Prints the output path and size.

    python scripts/pack_frontend.py [out.html]
"""

from __future__ import annotations

import base64
import mimetypes
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FE = ROOT / "frontend"


def data_uri(p: Path) -> str:
    mt = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    return f"data:{mt};base64," + base64.b64encode(p.read_bytes()).decode()


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "frontend" / "drishti.standalone.html"
    html = (FE / "index.html").read_text()

    data_js = (FE / "data.js")
    if not data_js.exists():
        print("frontend/data.js missing — run build_frontend_assets.py first", file=sys.stderr)
        return 1
    js = data_js.read_text()

    # embed each asset the data.js references as a data URI
    for name in sorted({m.group(0) for m in re.finditer(r"assets/[\w.\-]+\.jpg", js)}):
        p = FE / name
        if p.exists():
            js = js.replace('"' + name + '"', '"' + data_uri(p) + '"')

    html = html.replace('<script src="data.js"></script>', f"<script>\n{js}\n</script>")
    # any remaining <img src="assets/..."> in the shell (none expected, but be safe)
    html = re.sub(r'src="(assets/[\w.\-]+)"',
                  lambda m: f'src="{data_uri(FE / m.group(1))}"' if (FE / m.group(1)).exists() else m.group(0),
                  html)

    out.write_text(html)
    print(f"{out}  ·  {len(html) / 1024 / 1024:.2f} MB  ·  assets/ refs left: {html.count('assets/')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
