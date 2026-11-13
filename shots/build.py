"""Build the shot table: one row per hit, with who hit it, where everyone stood, and where the ball went.

Usage: uv run python shots/build.py            (reads every batch/out/*_decoded.json; writes shots/out/shots.json)

Needs only the front stages: player tracks, the 2D ball track and decoded hits and bounces. No 3D ball fit.
Target of a shot = where the ball next touched something on the other side:
  "bounce"  the floor point of the next bounce on the opponents' side
  "volley"  the court position of the opponent who hit it before it bounced
  none      the next event was on the hitter's own side or there was none (fault, net, end of point, or a missed event)
Coordinates are flipped so the hitter is always at the near end (y < 22) and targets fall in y 22..44.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "fit3d"))
from physics import Camera, NET_Y  # noqa: E402

CAM = Camera(json.loads((ROOT / "calib/out/t270_calibration.json").read_text()))


def flip(xy, far):
    return [20 - xy[0], 44 - xy[1]] if far else [float(xy[0]), float(xy[1])]


def main():
    rows = []
    for path in sorted(glob.glob(str(ROOT / "batch/out/*_decoded.json"))):
        tag = Path(path).name.replace("_decoded.json", "")
        ev = json.loads(Path(path).read_text())["events"]
        seg = json.loads((ROOT / f"batch/out/{tag}_segments.json").read_text())
        tr = np.load(ROOT / f"batch/out/{tag}_tracks.npz")
        ground, kpts = tr["smooth"], tr["kpts"]
        keep = np.array(seg["keep"]); xy = np.array([[np.nan if v is None else v for v in p] for p in seg["xy"]])
        seen = np.flatnonzero(~np.isnan(xy[:, 0]))
        if not len(seen):
            continue

        def who(e):
            """Which of the two players on the hitting side: the one whose wrist is nearest the ball in the image."""
            fr = int(np.clip(round(e["f"]), 0, len(ground) - 1))
            i = int(np.argmin(np.abs(keep - e["f"]))); px = xy[seen[np.argmin(np.abs(seen - i))]]
            cand = (0, 1) if e["side"] == "near" else (2, 3)
            d = []
            for p in cand:
                w = kpts[fr, p, 9:11, :2]
                d.append(np.nanmin(np.linalg.norm(w - px, axis=1)) if not np.isnan(w).all() else 1e9)
            return cand[int(np.argmin(d))], fr, px

        n_hit = 0
        for k, e in enumerate(ev):
            if e["kind"] != "hit":
                continue
            p, fr, px = who(e)
            if np.isnan(ground[fr]).any():
                continue
            far = e["side"] == "far"
            pos = [flip(ground[fr, q], far) for q in range(4)]
            order = ([2, 3, 0, 1] if far else [0, 1, 2, 3])
            me = pos[p]; mate = pos[p ^ 1]
            opp = [pos[q] for q in order[2:]]
            contact = CAM.at_depth(px, ground[fr, p, 1])            # ball at the hitter's depth: gives contact height
            row = {"clip": tag, "t": round(e["f"] / 60, 2), "index_in_clip": n_hit, "hitter": int(p), "side": e["side"],
                   "hitter_xy": [round(v, 2) for v in me], "partner_xy": [round(v, 2) for v in mate],
                   "opponents_xy": [[round(v, 2) for v in o] for o in sorted(opp)], "contact_height_ft": round(float(np.clip(contact[2], 0, 9)), 2),
                   "after_bounce": bool(k > 0 and ev[k - 1]["kind"] == "bounce"), "score": e.get("score")}
            n_hit += 1
            nxt = ev[k + 1] if k + 1 < len(ev) else None
            row["target"], row["target_kind"], row["dt"] = None, "none", None
            if nxt is not None:
                row["dt"] = round((nxt["f"] - e["f"]) / 60, 2)
                if nxt["kind"] == "bounce" and nxt["side"] != e["side"]:
                    row["target"] = [round(v, 2) for v in flip(nxt["floor_ft"], far)]; row["target_kind"] = "bounce"
                elif nxt["kind"] == "hit" and nxt["side"] != e["side"]:
                    q, fr2, _ = who(nxt)
                    if not np.isnan(ground[fr2, q]).any():
                        row["target"] = [round(v, 2) for v in flip(ground[fr2, q], far)]; row["target_kind"] = "volley"
            rows.append(row)
    out = ROOT / "shots/out/shots.json"
    out.write_text(json.dumps(rows, indent=0))
    kinds = {k: sum(r["target_kind"] == k for r in rows) for k in ("bounce", "volley", "none")}
    print(f"{len(rows)} hits from {len({r['clip'] for r in rows})} clips; targets: {kinds}")


if __name__ == "__main__":
    main()
