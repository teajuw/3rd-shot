"""Run GVHMR (static-camera mode) on the four tracked players of one shot and time it.

Usage (from anywhere): bodies/.venv/bin/python bodies/run_gvhmr.py <video> <tracks.npz> <calibration.json> <out_prefix>
           [--step 2] [--frames 1800] [--device mps]

Same boxes, frames and focal length as run_sam3d.py, and the same output layout, so
render_bench.py can read either. GVHMR is CUDA-only upstream; here every .cuda() is
redirected to the chosen device and pytorch3d is replaced by bodies/shims.
"""
import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
GV = ROOT / "vendor/GVHMR"
sys.path.insert(0, str(ROOT / "bodies/shims"))
sys.path.insert(0, str(GV))
sys.path.insert(0, str(ROOT / "bodies"))
from run_sam3d import boxes_from_kpts  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("video"); ap.add_argument("tracks"); ap.add_argument("calib"); ap.add_argument("out")
ap.add_argument("--step", type=int, default=2); ap.add_argument("--frames", type=int, default=1800)
ap.add_argument("--device", default="mps")
a = ap.parse_args()
a.video, a.tracks, a.calib, a.out = (str(Path(p).resolve()) for p in (a.video, a.tracks, a.calib, a.out))
os.chdir(GV)                                   # GVHMR uses paths relative to its repo
DEV = torch.device(a.device)

# --- make the CUDA-only code run on DEV ---
torch.Tensor.cuda = lambda self, *x, **k: self.to(DEV)
torch.nn.Module.cuda = lambda self, *x, **k: self.to(DEV)
try:
    from lightning_fabric.utilities.device_dtype_mixin import _DeviceDtypeModuleMixin as _M
    _M.cuda = lambda self, *x, **k: self.to(DEV)
except ImportError:
    pass
_load = torch.load
torch.load = lambda *x, **k: _load(*x, **{"weights_only": False, **k})      # official checkpoints, pickled
_w = types.ModuleType("wis3d"); _w.Wis3D = object; sys.modules["wis3d"] = _w    # visualiser used only in training code
pre = types.ModuleType("hmr4d.utils.preproc"); pre.__path__ = [str(GV / "hmr4d/utils/preproc")]
sys.modules["hmr4d.utils.preproc"] = pre       # skip its __init__, which pulls in the tracker and SLAM

from hydra import compose, initialize_config_module  # noqa: E402
import hydra  # noqa: E402
import hmr4d.configs  # noqa: E402,F401
from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy  # noqa: E402
from hmr4d.utils.geo_transform import compute_cam_angvel  # noqa: E402
from hmr4d.utils.preproc.vitfeat_extractor import Extractor, get_batch  # noqa: E402
from hmr4d.utils.preproc.vitpose import VitPoseExtractor  # noqa: E402
from hmr4d.utils.smplx_utils import make_smplx  # noqa: E402


def sync():
    if DEV.type == "mps":
        torch.mps.synchronize()


def main():
    tr = np.load(a.tracks)
    kpts, start = tr["kpts"], int(tr["start_frame"])
    cal = json.loads(Path(a.calib).read_text())
    e = cal["no_distortion"]
    e = eval(e) if isinstance(e, str) else e  # noqa: S307
    W, H = cal["image_size"]
    idx = list(range(0, min(a.frames, len(kpts)), a.step))
    n, P = len(idx), kpts.shape[1]
    T = {}

    # boxes: fill gaps so every player has a box on every frame (GVHMR wants a full sequence)
    boxes = np.stack([boxes_from_kpts(kpts[i], W, H) for i in idx])          # n, P, 4
    valid = ~np.isnan(boxes[..., 0])
    for p in range(P):
        ok = valid[:, p]
        for c in range(4):
            boxes[:, p, c] = np.interp(np.arange(n), np.flatnonzero(ok), boxes[ok, p, c])
    bbx = [get_bbx_xys_from_xyxy(torch.from_numpy(boxes[:, p]).float(), base_enlarge=1.2).float() for p in range(P)]

    # crops, 256x256, one pass over the video
    t0 = time.time()
    cap = cv2.VideoCapture(a.video); cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    crops = [[] for _ in range(P)]; bbx2 = [[] for _ in range(P)]
    cur = 0
    for j, i in enumerate(idx):
        while cur <= i:
            ok, frame = cap.read(); cur += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[None]
        for p in range(P):
            im, b2 = get_batch(rgb, bbx[p][j:j + 1], img_ds=1.0, path_type="np")
            crops[p].append(im); bbx2[p].append(b2)
    crops = [torch.cat(c).float() for c in crops]; bbx2 = [torch.cat(b).float() for b in bbx2]
    T["crop_s"] = time.time() - t0

    t0 = time.time()
    vp = VitPoseExtractor(tqdm_leave=False)
    kp2d = [vp.extract(crops[p], bbx2[p]) for p in range(P)]
    sync(); T["vitpose_s"] = time.time() - t0
    del vp

    t0 = time.time()
    ex = Extractor(tqdm_leave=False)
    feats = [ex.extract_video_features(crops[p], bbx2[p]) for p in range(P)]
    sync(); T["hmr2_features_s"] = time.time() - t0
    del ex, crops

    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        # the three registrations inference needs (register_store_gvhmr also imports datasets and renderers)
        import hmr4d.model.gvhmr.gvhmr_pl_demo, hmr4d.model.gvhmr.utils.endecoder, hmr4d.network.gvhmr.relative_transformer  # noqa: F401,E401
        cfg = compose(config_name="demo", overrides=["video_name=bench", "static_cam=True"])
    t0 = time.time()
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    model.load_pretrained_model(cfg.ckpt_path)
    model = model.eval().to(DEV)
    K = torch.tensor([[e["f_px"], 0, W / 2], [0, e["f_px"], H / 2], [0, 0, 1]], dtype=torch.float32).repeat(n, 1, 1)
    preds = []
    for p in range(P):
        data = {"length": torch.tensor(n), "bbx_xys": bbx2[p], "kp2d": kp2d[p].float(), "K_fullimg": K,
                "cam_angvel": compute_cam_angvel(torch.eye(3).repeat(n, 1, 1)), "f_imgseq": feats[p]}
        preds.append(model.predict(data, static_cam=True))
    sync(); T["gvhmr_s"] = time.time() - t0

    t0 = time.time()
    smplx = make_smplx("supermotion").to(DEV)
    faces = np.asarray(smplx.faces, np.int64)
    verts = None
    res = {k: np.zeros((n, P) + s, np.float32) for k, s in
           {"pred_keypoints_3d": (70, 3), "pred_cam_t": (3,), "shape_params": (10,), "scale_params": (1,),
            "body_pose": (63,), "global_orient": (3,), "transl": (3,), "global_orient_world": (3,), "transl_world": (3,), "body_pose_world": (63,)}.items()}
    for p in range(P):
        inc = {k: v.to(DEV).float() for k, v in preds[p]["smpl_params_incam"].items()}
        out = smplx(**inc)
        v = out.vertices.cpu().numpy(); J = out.joints.cpu().numpy()
        if verts is None:
            verts = np.lib.format.open_memmap(a.out + "_verts.npy", mode="w+", dtype=np.float16, shape=(n, P) + v.shape[1:])
        verts[:, p] = v
        # same slots as SAM 3D Body's 70 keypoints for the joints the benchmark reads: hips 9/10, ankles 13/14, wrists 41/62
        for slot, jid in ((9, 1), (10, 2), (13, 7), (14, 8), (62, 20), (41, 21)):
            res["pred_keypoints_3d"][:, p, slot] = J[:, jid]
        g = preds[p]["smpl_params_global"]
        for k_out, src, k_in in (("shape_params", inc, "betas"), ("body_pose", inc, "body_pose"), ("global_orient", inc, "global_orient"), ("transl", inc, "transl"),
                                 ("global_orient_world", g, "global_orient"), ("transl_world", g, "transl"), ("body_pose_world", g, "body_pose")):
            res[k_out][:, p] = src[k_in].detach().cpu().numpy().reshape(n, -1)
    verts.flush()
    T["smplx_s"] = time.time() - t0

    total = sum(T.values())
    secs = np.full(n, total / n)
    np.savez_compressed(a.out + ".npz", frames=np.array(idx), start_frame=start, valid=valid, boxes=boxes.astype(np.float32), secs=secs,
                        faces=faces, mode="gvhmr", device=a.device, timing=json.dumps(T), **res)
    print(json.dumps({k: round(v, 1) for k, v in T.items()}))
    print(f"DONE {n} frames x {P} players; {total:.1f} s total, {total / (n * P):.3f} s per person-frame")


if __name__ == "__main__":
    main()
