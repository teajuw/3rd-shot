"""Cut a window out of a browser export (bodies/export_view.py) for the article's live demo.

Usage: uv run python bodies/export_demo.py <view_dir> <out_dir> [t0=20] [t1=30]
Reads <view_dir>/meta.json, near.bin and far.bin; writes the same three files to <out_dir>, holding only t0..t1 seconds.
"""
import json
import sys
from pathlib import Path

import numpy as np

src, out = Path(sys.argv[1]), Path(sys.argv[2])
t0 = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0
t1 = float(sys.argv[4]) if len(sys.argv) > 4 else 30.0
m = json.loads((src / "meta.json").read_text())
K, NV, fps = m["frames"], m["verts"], m["fps"]
k0, k1 = int(round(t0 * fps)), min(K, int(round(t1 * fps)))
out.mkdir(parents=True, exist_ok=True)
for name in ("near", "far"):
    a = np.fromfile(src / f"{name}.bin", "<i2").reshape(2, K, NV, 3)
    a[:, k0:k1].tofile(out / f"{name}.bin")
m["frames"] = k1 - k0
m["start_s"] = m["start_s"] + k0 / fps
m["roots"] = [r[k0:k1] for r in m["roots"]]
m["ball"] = m["ball"][k0:k1] if m["ball"] else None
m["paddle"] = [p[k0:k1] for p in m["paddle"]] if m["paddle"] else None
m["hits"] = [{"t": round(h["t"] - t0, 2), "player": h["player"]} for h in (m["hits"] or []) if t0 <= h["t"] < t1]
(out / "meta.json").write_text(json.dumps(m, separators=(",", ":")))
print(m["frames"], "frames from", round(m["start_s"], 2), "s;", {f.name: f.stat().st_size for f in sorted(out.iterdir())})
