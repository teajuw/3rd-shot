"""Per-frame ball candidates from two zero-training detectors, for one baseline-camera shot.

A. colour+motion: subtract the shot's median background, keep small blobs whose change is
   "yellow-green appearing" (red and green up, blue down, not skin). Pure NumPy/OpenCV.
B. TrackNetV3 from RacketVision (MIT), trained on tennis, badminton and table tennis:
   background + 4 frames at 512x288 -> heatmap. Run with frame stride 1 and 2.

Usage: python candidates.py <video> <start_frame> <n_frames> <out.npz>
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "players"))
from track_players import frames  # noqa: E402

SIZE = (1920, 1080)
MAXC = 6  # candidates kept per frame


# ---------- A. colour + motion ----------
def colour_candidates(f, bg):
    d = f.astype(np.int16) - bg
    dB, dG, dR = d[..., 0], d[..., 1], d[..., 2]
    B, G, R = (f[..., i].astype(np.int16) for i in range(3))
    change = np.abs(d).sum(axis=2)
    appear = (dB < -20) & (dG > 15) & (dR > 20) & (dG - dR > -40)        # yellow-green added over the background
    chroma = (G - B > 45) & (R - B > 25) & (np.abs(G - R) < 70) & (G > 110) & (change > 60)  # pixel itself is ball-coloured
    mask = (appear | chroma).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask, connectivity=8)
    yellow = np.clip(np.minimum(R, G) - B, 0, None).astype(np.float32)
    out = []
    for i in range(1, n):
        area = st[i, cv2.CC_STAT_AREA]
        w, h = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        if area < 4 or area > 900 or max(w, h) > 90:
            continue
        x, y = st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP]
        m = lab[y:y + h, x:x + w] == i
        score = float(yellow[y:y + h, x:x + w][m].mean())
        if float((G[y:y + h, x:x + w][m] - R[y:y + h, x:x + w][m]).mean()) < -8:
            continue  # redder than green: skin, not a yellow-green ball
        out.append((cen[i, 0], cen[i, 1], score, area, max(w, h)))
    out.sort(key=lambda c: -c[2] * min(c[3], 40))
    return out[:MAXC]


# ---------- B. TrackNetV3 (RacketVision weights) ----------
class Conv2DBlock(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.conv = nn.Conv2d(i, o, 3, padding="same", bias=False)
        self.bn = nn.BatchNorm2d(o)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Double2DConv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.conv_1, self.conv_2 = Conv2DBlock(i, o), Conv2DBlock(o, o)

    def forward(self, x):
        return self.conv_2(self.conv_1(x))


class Triple2DConv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.conv_1, self.conv_2, self.conv_3 = Conv2DBlock(i, o), Conv2DBlock(o, o), Conv2DBlock(o, o)

    def forward(self, x):
        return self.conv_3(self.conv_2(self.conv_1(x)))


class TrackNet(nn.Module):
    def __init__(self, in_dim=15, out_dim=4, d=64):
        super().__init__()
        self.down_block_1 = Double2DConv(in_dim, d)
        self.down_block_2 = Double2DConv(d, d * 2)
        self.down_block_3 = Triple2DConv(d * 2, d * 4)
        self.bottleneck = Triple2DConv(d * 4, d * 8)
        self.up_block_1 = Triple2DConv(d * 12, d * 4)
        self.up_block_2 = Double2DConv(d * 6, d * 2)
        self.up_block_3 = Double2DConv(d * 3, d)
        self.predictor = nn.Conv2d(d, out_dim, 1)

    def forward(self, x):
        pool, up = nn.functional.max_pool2d, nn.functional.interpolate
        x1 = self.down_block_1(x)
        x2 = self.down_block_2(pool(x1, 2))
        x3 = self.down_block_3(pool(x2, 2))
        x = self.bottleneck(pool(x3, 2))
        x = self.up_block_1(torch.cat([up(x, scale_factor=2), x3], 1))
        x = self.up_block_2(torch.cat([up(x, scale_factor=2), x2], 1))
        x = self.up_block_3(torch.cat([up(x, scale_factor=2), x1], 1))
        return torch.sigmoid(self.predictor(x))[:, -1]   # heatmap for the last frame of the stack


class _Stub:
    """Stands in for any non-tensor class stored in the checkpoint (training metadata)."""
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        pass

    def __call__(self, *a, **k):
        return _Stub()


class _SafePickle:
    """Pickle module for torch.load that only resolves torch/numpy/builtin container classes.
    The checkpoint is a downloaded file that references mmengine classes; everything outside
    the allowlist becomes an inert stub, so nothing from the file is imported or executed."""
    import pickle as _p
    __name__ = "safe_pickle"
    load = _p.load

    class Unpickler(_p.Unpickler):
        ALLOWED = ("torch", "numpy", "collections", "builtins", "_codecs")

        def find_class(self, module, name):
            if module.split(".")[0] in self.ALLOWED and name not in ("eval", "exec", "compile", "open", "__import__", "getattr", "system"):
                return super().find_class(module, name)
            return _Stub


def load_tracknet(path):
    state = torch.load(path, map_location="cpu", weights_only=False, pickle_module=_SafePickle)
    state = state.get("state_dict", state)
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    m = TrackNet()
    missing, unexpected = m.load_state_dict(state, strict=False)
    assert not missing, missing
    return m.to("mps").eval()


def heatmap_peaks(hm, thr=0.5):
    mask = (hm > thr).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h = st[i, 0], st[i, 1], st[i, 2], st[i, 3]
        conf = float(hm[y:y + h, x:x + w][lab[y:y + h, x:x + w] == i].mean())
        out.append((cen[i, 0] * SIZE[0] / 512, cen[i, 1] * SIZE[1] / 288, conf, st[i, cv2.CC_STAT_AREA], max(w, h)))
    out.sort(key=lambda c: -c[2] * c[3])
    return out[:MAXC]


def tracknet_candidates(model, small, bg_small, stride):
    """small: (N,288,512,3) float32 BGR in [0,1]."""
    n = len(small)
    res = []
    bgc = np.moveaxis(bg_small, -1, 0)
    for s in range(0, n, 16):
        batch = []
        for i in range(s, min(n, s + 16)):
            ids = [max(0, i - stride * k) for k in (3, 2, 1, 0)]
            batch.append(np.concatenate([bgc] + [np.moveaxis(small[j], -1, 0) for j in ids], 0))
        with torch.no_grad():
            hm = model(torch.from_numpy(np.array(batch)).to("mps")).cpu().numpy()
        res += [heatmap_peaks(h) for h in hm]
    return res


def pack(cands):
    a = np.full((len(cands), MAXC, 5), np.nan, np.float32)
    for i, cs in enumerate(cands):
        for j, c in enumerate(cs):
            a[i, j] = c
    return a


def main():
    video, start, n, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    fr = np.stack([f.copy() for f in frames(video, start, n, SIZE)])
    n = len(fr)
    bg = np.median(fr[:: max(1, n // 40)], axis=0).astype(np.int16)
    cv2.imwrite(out.replace(".npz", "_bg.jpg"), bg.astype(np.uint8))

    t0 = time.time()
    colour = [colour_candidates(f, bg) for f in fr]
    t_colour = time.time() - t0

    small = np.stack([cv2.resize(f, (512, 288)) for f in fr]).astype(np.float32) / 255.0
    bg_small = cv2.resize(bg.astype(np.uint8), (512, 288)).astype(np.float32) / 255.0
    model = load_tracknet(str(Path(__file__).resolve().parent.parent / "models/balltrack_best.pth"))
    t0 = time.time()
    tn1 = tracknet_candidates(model, small, bg_small, 1)
    t_tn = time.time() - t0
    tn2 = tracknet_candidates(model, small, bg_small, 2)

    np.savez(out, colour=pack(colour), tracknet_s1=pack(tn1), tracknet_s2=pack(tn2), start_frame=start)
    for name, c, t in (("colour+motion", colour, t_colour), ("tracknet stride 1", tn1, t_tn), ("tracknet stride 2", tn2, t_tn)):
        k = np.array([len(x) for x in c])
        print(f"{name:<18} frames with >=1 candidate: {100 * (k > 0).mean():5.1f}%  mean candidates/frame {k.mean():.2f}  {1000 * t / n:.1f} ms/frame")


if __name__ == "__main__":
    main()
