"""Paddle v1: a rigid paddle in each player's hand, with the face turned to meet the ball at hits.

Usage: bodies/.venv/bin/python bodies/paddle.py <grounded_prefix> <out_prefix> <fit_prefix> [<fit_prefix> ...]
  fit prefixes are fit3d outputs (<prefix>_path.npz and <prefix>_fit.json), highest priority first;
  where two cover the same time the earlier one wins.

1. Ball and events for the whole shot are merged from the fits.
2. Hand: for each player, the hand whose wrist is nearer the ball at that player's hits.
3. Between hits (a default, not a measurement): the handle continues the forearm with the wrist
   cocked 30 degrees upward, and the face turns as far towards the net as that handle allows.
4. At each hit the face is turned to the direction the ball's velocity changed (a ball bouncing
   off a flat face without friction leaves along the face normal).
5. The face reaches the hit direction by the shortest turn and carries the handle with it, held
   for +-0.05 s and faded over +-0.30 s; the whole path is then low-passed at 8 Hz.
Writes <out_prefix>.npz (ball, per-frame paddle centre/axis/normal, hits) and <out_prefix>_stats.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

HAND = {"right": dict(wrist=41, elbow=8, index=28, pinky=40), "left": dict(wrist=62, elbow=7, index=49, pinky=61)}
LENGTH, GRIP_FROM_BUTT = 1.31, 0.22          # ft: 15.75 in overall; palm centre sits 2.6 in up the handle
BLEND_S, HOLD_S = 0.30, 0.05                # a hit holds the face for +-0.05 s and fades out by +-0.30 s
HANDLE_HZ = 3.0                              # filter on the default face direction between hits
COCK_DEG = 30                                # default angle between forearm and handle (wrist cocked)


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def rot_between(a, b):
    """Rotation matrix taking unit a to unit b by the shortest turn."""
    v, c = np.cross(a, b), float(a @ b)
    if c > 1 - 1e-9:
        return np.eye(3)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k / (1 + c)


def scale_rot(Rm, w):
    """Rm applied by a fraction w of its angle."""
    ang = np.arccos(np.clip((np.trace(Rm) - 1) / 2, -1, 1))
    if ang < 1e-6 or w <= 0:
        return np.eye(3)
    ax = unit(np.array([Rm[2, 1] - Rm[1, 2], Rm[0, 2] - Rm[2, 0], Rm[1, 0] - Rm[0, 1]]))
    k = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(ang * w) * k + (1 - np.cos(ang * w)) * k @ k


def lowpass(x, fc, fps, med=3):
    from scipy.signal import butter, filtfilt, medfilt
    x = np.asarray(x, np.float64)
    if med > 1:
        x = medfilt(x, [med] + [1] * (x.ndim - 1))
    b, a = butter(2, fc / (fps / 2))
    return np.ascontiguousarray(filtfilt(b, a, x, axis=0))


def seg_dist(pts, a, b):
    """pts (n, m, 3), a and b (n, 3): distance to the segment and the unit direction away from it."""
    ab = b - a
    t = np.clip(((pts - a[:, None]) * ab[:, None]).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9)[:, None], 0, 1)
    d = pts - (a[:, None] + t[..., None] * ab[:, None])
    dist = np.linalg.norm(d, axis=-1)
    return dist, d / np.maximum(dist, 1e-9)[..., None]


def body_penetration(kp, grip, u, nn, H):
    """How far the paddle face cuts into the body (ft, per frame), and which way to lean the handle to get out.
    The body is a few capsules through the joints; the paddle arm is left out."""
    side = np.cross(u, nn)
    reach = LENGTH - GRIP_FROM_BUTT
    pts = np.stack([grip + reach * u, grip + 0.65 * reach * u, grip + 0.65 * reach * u + 0.3 * side, grip + 0.65 * reach * u - 0.3 * side, grip + 0.35 * reach * u], 1)
    mid = lambda a, b: (kp[:, a] + kp[:, b]) / 2  # noqa: E731
    other = "left" if H is HAND["right"] else "right"
    caps = [(mid(5, 6), mid(9, 10), 0.55), (kp[:, 9], kp[:, 11], 0.30), (kp[:, 10], kp[:, 12], 0.30), (kp[:, 11], kp[:, 13], 0.22), (kp[:, 12], kp[:, 14], 0.22),
            (kp[:, 0], mid(5, 6), 0.38), (kp[:, HAND[other]["elbow"]], kp[:, HAND[other]["wrist"]], 0.17)]
    pen = np.zeros(len(kp)); push = np.zeros((len(kp), 3))
    for a, b, r in caps:
        dist, away = seg_dist(pts, a, b)
        depth = np.clip(r - dist, 0, None)                         # n, m
        k = depth.argmax(1); best = depth.max(1)
        better = best > pen
        pen[better] = best[better]; push[better] = (away[np.arange(len(kp)), k] * best[:, None])[better]
    return pen, push


def motion_stats(axis, normal, fps):
    """Turning speed of the paddle and the number of sudden flips of the face."""
    out = {"turn_speed_deg_s_p99": [], "turn_speed_deg_s_max": [], "face_flips": []}
    for p in range(axis.shape[1]):
        F0 = np.stack([axis[:-1, p], normal[:-1, p], np.cross(axis[:-1, p], normal[:-1, p])], 2)
        F1 = np.stack([axis[1:, p], normal[1:, p], np.cross(axis[1:, p], normal[1:, p])], 2)
        tr = np.einsum("nij,nij->n", F0, F1)
        ang = np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))) * fps
        k = max(int(round(0.1 * fps)), 1)
        flip = (normal[:-k, p] * normal[k:, p]).sum(1) < 0          # face turned more than 90 degrees within 0.1 s
        out["turn_speed_deg_s_p99"].append(round(float(np.percentile(ang, 99)))); out["turn_speed_deg_s_max"].append(round(float(ang.max())))
        out["face_flips"].append(int((np.diff(flip.astype(int)) == 1).sum() + int(flip[0])))
    return out


def main():
    g = np.load(sys.argv[1] + ".npz")
    out = sys.argv[2]
    frames, fps = g["frames"].astype(float), float(g["fps"])      # shot frames at 60 fps
    n = len(frames)
    kp = g["kp_world"]                                             # n, P, 70, 3 (court feet)
    P = kp.shape[1]

    # ---- 1. merge ball and events ----
    ball = np.full((n, 3), np.nan); src = np.full(n, -1)
    hits = []
    for pri, pre in enumerate(sys.argv[3:]):
        b = np.load(pre + "_path.npz"); fit = json.loads(Path(pre + "_fit.json").read_text())
        path, keep = b["path"], b["keep"].astype(float)
        ok = ~np.isnan(path[:, 0])
        for c in range(3):
            v = np.interp(frames, keep[ok], path[ok, c], left=np.nan, right=np.nan)
            # do not bridge gaps inside the fit
            gap = np.interp(frames, keep, (~ok).astype(float)) > 0.5
            v[gap] = np.nan
            take = np.isnan(ball[:, c]) & ~np.isnan(v) & ((src == -1) | (src == pri))
            ball[take, c] = v[take]
        src[(src == -1) & ~np.isnan(ball[:, 0])] = pri
        for e in fit["events"]:
            if e["kind"] != "hit":
                continue
            # contact time and place as the fit solved them (the event's sample index can be 60 ms and a foot or two away)
            u_t = float(e["t"]) * 50.0                              # position on the 50 Hz unique-frame clock
            f = float(np.interp(u_t, np.arange(len(keep)), keep))
            j = int(np.argmin(np.abs(frames - f)))
            if src[j] != pri:
                continue                                           # a higher-priority fit owns this time
            lo_i, hi_i = int(np.floor(u_t - 0.5)), int(np.ceil(u_t + 0.5))

            def vel(a, b):
                a, b = max(a, 0), min(b, len(path) - 1)
                if b <= a or np.isnan(path[[a, b], 0]).any():
                    return None
                return (path[b] - path[a]) / ((keep[b] - keep[a]) / 60.0)
            hits.append({"frame": f, "t": f / 60.0, "player": int(e["player"]), "pos": [float(x) for x in e["pos_ft"]],
                         "v_in": vel(lo_i - 2, lo_i), "v_out": vel(hi_i, hi_i + 2), "fit": Path(pre).name})
    hits.sort(key=lambda h: h["t"])

    # ---- 2. which hand ----
    hand = []
    for p in range(P):
        votes = {"right": 0, "left": 0}
        for h in hits:
            if h["player"] != p:
                continue
            j = int(np.argmin(np.abs(frames - h["frame"])))
            d = {s: np.linalg.norm(kp[j, p, HAND[s]["wrist"]] - np.array(h["pos"])) for s in HAND}
            votes[min(d, key=d.get)] += 1
        hand.append("left" if votes["left"] > votes["right"] else "right")

    # ---- 3 and 4. one smooth paddle path per player ----
    # The body model's finger and wrist-roll estimates did not predict the face at hits on this footage,
    # so between hits the pose is a default built from what is reliable: the forearm, and the fact that a
    # paddle generally faces the net. Hits pull the face to the ball-derived direction. Everything is then
    # filtered as one path, so the paddle cannot jump, and the same side always faces the net (a paddle is
    # symmetric, so forehand and backhand need no flip).
    centre = np.zeros((n, P, 3)); axis = np.zeros((n, P, 3)); normal = np.zeros((n, P, 3)); grip_pt = np.zeros((n, P, 3))
    up = np.array([0.0, 0.0, 1.0])
    stats_hits, clip_before, clip_after, wrist_all, fore_speed = [], [], [], [], []
    for p in range(P):
        H = HAND[hand[p]]
        W, E = kp[:, p, H["wrist"]], kp[:, p, H["elbow"]]
        I, K = kp[:, p, H["index"]], kp[:, p, H["pinky"]]
        fa = unit(W - E)
        grip = (W + I + K) / 3                                     # in the palm, not at the wrist
        to_net = np.array([0.0, 1.0 if np.median(W[:, 1]) < 22 else -1.0, 0.0])
        face = unit(to_net + 0.25 * up)                            # facing the net, slightly open
        # default: the handle continues the forearm, cocked towards "up" by COCK_DEG; the face turns
        # as far towards the net as that handle allows. The handle has priority, so the paddle moves with the arm.
        lift = unit(up - (fa @ up)[:, None] * fa)
        u_d = unit(np.cos(np.radians(COCK_DEG)) * fa + np.sin(np.radians(COCK_DEG)) * lift)
        u_d = unit(lowpass(u_d, 6.0, fps))
        # Roll about the handle. Carry a reference direction along the handle's path without twisting it,
        # and describe the face by one angle phi from that reference. Between hits phi is pulled gently
        # towards "facing the net" (only where that is well defined); at hits it is pinned to the ball's
        # direction; and it is solved for the whole rally at once so it cannot spin or flip.
        e = np.zeros((n, 3)); e0 = face - (u_d[0] @ face) * u_d[0]
        e[0] = unit(e0 if np.linalg.norm(e0) > 0.1 else np.cross(u_d[0], up))
        for i in range(n - 1):
            v = rot_between(u_d[i], u_d[i + 1]) @ e[i]
            e[i + 1] = unit(v - (v @ u_d[i + 1]) * u_d[i + 1])
        f = np.cross(u_d, e)
        nd_raw = face - (u_d @ face)[:, None] * u_d
        T = np.arctan2((nd_raw * f).sum(1), (nd_raw * e).sum(1)); Wt = 0.02 * (nd_raw ** 2).sum(1)
        for h in hits:
            if h["player"] != p or h["v_in"] is None or h["v_out"] is None:
                continue
            j = int(np.argmin(np.abs(frames - h["frame"])))
            dv = np.array(h["v_out"]) - np.array(h["v_in"])
            if np.linalg.norm(dv) <= 5 or np.linalg.norm(np.array(h["pos"]) - grip[j]) >= 3.0:
                continue
            tg = unit(dv)
            for jj in np.flatnonzero(np.abs(frames - h["frame"]) / 60.0 <= HOLD_S):
                pr = tg - (tg @ u_d[jj]) * u_d[jj]
                T[jj] = np.arctan2(pr @ f[jj], pr @ e[jj]); Wt[jj] = 10.0 * (pr @ pr)
        rep = np.zeros(n); cur = T[int(np.argmax(Wt > 1e-4))]
        for i in range(n):                                         # the two faces are the same paddle: angles are equal modulo 180 degrees
            rep[i] = T[i] + np.pi * np.round((cur - T[i]) / np.pi)
            cur += min(1.0, 5.0 * Wt[i]) * (rep[i] - cur)
        D1 = np.diff(np.eye(n), 1, axis=0); D2 = np.diff(np.eye(n), 2, axis=0)
        phi = np.linalg.solve(np.diag(Wt) + 0.5 * D1.T @ D1 + 5.0 * D2.T @ D2, Wt * rep)
        n_d = np.cos(phi)[:, None] * e + np.sin(phi)[:, None] * f
        F = np.stack([u_d, n_d, np.cross(u_d, n_d)], 2)            # n, 3, 3 (columns: handle, face normal, side)
        weight = np.zeros(n)
        for h in hits:
            if h["player"] != p:
                continue
            j = int(np.argmin(np.abs(frames - h["frame"])))
            b = np.array(h["pos"]); reach = float(np.linalg.norm(b - grip[j]))
            rec = {"t": round(float(h["t"]), 2), "player": p, "fit": h["fit"], "ball_to_grip_ft": round(reach, 2), "frame_index": j}
            if h["v_in"] is not None and h["v_out"] is not None and reach < 3.0:
                dv = np.array(h["v_out"]) - np.array(h["v_in"])
                if np.linalg.norm(dv) > 5:                         # ft/s; below this the direction is noise
                    target = unit(dv)
                    if target @ n_d[j] < 0:
                        target = -target                           # a paddle is symmetric: use whichever side is already nearer
                    rec.update(target=target.tolist(), default_face_vs_ball_deg=round(float(np.degrees(np.arccos(np.clip(target @ n_d[j], -1, 1)))), 1),
                               ball_off_face_plane_ft=round(float(abs((b - grip[j]) @ target)), 2))
                    dt = np.abs(frames - h["frame"]) / 60.0
                    w = np.where(dt <= HOLD_S, 1.0, 0.5 + 0.5 * np.cos(np.pi * np.clip((dt - HOLD_S) / (BLEND_S - HOLD_S), 0, 1)))
                    # then, if the ball is within the paddle's length of the palm and close to the face plane, swing the
                    # handle within that plane to point at the contact, provided the wrist can bend that far
                    Fj = rot_between(n_d[j], target) @ np.stack([u_d[j], n_d[j], np.cross(u_d[j], n_d[j])], 1)
                    want = b - grip[j]; off = want @ target; inpl = want - off * target
                    aim = np.eye(3)
                    if abs(off) < 0.5 and 0.25 < np.linalg.norm(inpl) < LENGTH - GRIP_FROM_BUTT + 0.1:
                        u_h = unit(inpl)
                        if np.degrees(np.arccos(np.clip(u_h @ fa[j], -1, 1))) <= 110:
                            aim = rot_between(Fj[:, 0], u_h); rec["handle_aimed_at_ball"] = True
                    for jj in np.flatnonzero(w > weight):
                        # the face goes to the ball's direction by the shortest turn; the handle is carried along with it
                        F[jj] = scale_rot(aim, float(w[jj])) @ scale_rot(rot_between(n_d[jj], target), float(w[jj])) @ np.stack([u_d[jj], n_d[jj], np.cross(u_d[jj], n_d[jj])], 1)
                        weight[jj] = w[jj]
                    rec["wrist_angle_at_hit_deg"] = round(float(np.degrees(np.arccos(np.clip(F[j][:, 0] @ fa[j], -1, 1)))), 1)
                    rec["ball"] = b.tolist()
            stats_hits.append(rec)
        six = lowpass(np.concatenate([F[:, :, 0], F[:, :, 1]], 1), 8.0, fps, med=1)
        u = unit(six[:, :3]); nn = unit(six[:, 3:] - (six[:, 3:] * u).sum(1, keepdims=True) * u)
        pen, _ = body_penetration(kp[:, p], grip, u, nn, H)
        clip_before.append(pen > 0.05); clip_after.append(pen > 0.05)
        wrist_all.append(np.degrees(np.arccos(np.clip((u * fa).sum(1), -1, 1))))
        fore_speed.append(np.degrees(np.arccos(np.clip((fa[:-1] * fa[1:]).sum(1), -1, 1))) * fps)
        axis[:, p], normal[:, p], grip_pt[:, p] = u, nn, grip
        centre[:, p] = grip + (LENGTH / 2 - GRIP_FROM_BUTT) * u
    prior_normal = normal
    for r in stats_hits:
        if "target" in r:
            # how far the ball is from the paddle face (a 0.89 x 0.66 ft rectangle starting 0.2 ft up from the palm)
            j, p = r["frame_index"], r["player"]
            rel = np.array(r.pop("ball")) - grip_pt[j, p]
            side = np.cross(axis[j, p], normal[j, p])
            a_, s_, n_ = rel @ axis[j, p], rel @ side, rel @ normal[j, p]
            r["ball_to_face_ft"] = round(float(np.sqrt(max(0.2 - a_, 0, a_ - 1.09) ** 2 + max(abs(s_) - 0.33, 0) ** 2 + n_ ** 2)), 2)
            r["face_error_after_smoothing_deg"] = round(float(np.degrees(np.arccos(np.clip(np.array(r["target"]) @ normal[r["frame_index"], r["player"]], -1, 1)))), 1)
            del r["target"]

    np.savez_compressed(out + ".npz", frames=g["frames"], ball=ball, ball_src=src, centre=centre, axis=axis, normal=normal, prior_normal=prior_normal,
                        hand=np.array(hand), hits=json.dumps(hits, default=lambda x: None if x is None else list(map(float, x))))
    used = [r for r in stats_hits if "default_face_vs_ball_deg" in r]
    med = lambda k, rows=used: round(float(np.median([r[k] for r in rows])), 2) if rows else None  # noqa: E731
    motion = motion_stats(axis, normal, fps)
    stats = {"ball_coverage_pct": round(100 * float((~np.isnan(ball[:, 0])).mean()), 1), "ball_seconds": round(float((~np.isnan(ball[:, 0])).sum() / fps), 1),
             "hand": hand, "hits": len(hits), "hits_used": len(used),
             "default_face_vs_ball_deg_median": med("default_face_vs_ball_deg"), "face_error_after_smoothing_deg_median": med("face_error_after_smoothing_deg"),
             "face_error_after_smoothing_deg_max": round(float(max(r["face_error_after_smoothing_deg"] for r in used)), 1) if used else None,
             "ball_to_grip_ft_median": med("ball_to_grip_ft", stats_hits), "ball_to_face_ft_median": med("ball_to_face_ft"), "ball_touching_face": f"{sum(r['ball_to_face_ft'] <= 0.25 for r in used)} of {len(used)}",
             "handles_aimed_at_ball": int(sum(r.get("handle_aimed_at_ball", False) for r in used)), "ball_off_face_plane_ft_median": med("ball_off_face_plane_ft"),
             "wrist_angle_deg_p95": [round(float(np.percentile(w, 95))) for w in wrist_all], "wrist_angle_deg_max": [round(float(w.max())) for w in wrist_all],
             "forearm_turn_speed_deg_s_p99": [round(float(np.percentile(f, 99))) for f in fore_speed],
             "wrist_angle_at_hit_deg_median": med("wrist_angle_at_hit_deg"),
             "frames_inside_body_pct": [round(100 * float(c.mean()), 1) for c in clip_after], **motion, "per_hit": stats_hits}
    Path(out + "_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k != "per_hit"}, indent=1))


if __name__ == "__main__":
    main()
