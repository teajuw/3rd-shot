"""Rebuild SAM 3D Body meshes and joints from its parameters (Meta's MHR rig), on the CPU."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "vendor/sam-3d-body"))
from sam_3d_body import load_sam_3d_body  # noqa: E402
from sam_3d_body.models.modules.mhr_utils import compact_cont_to_model_params_body, compact_model_params_to_cont_body  # noqa: E402,F401

_head = None


def head():
    global _head
    if _head is None:
        model, _ = load_sam_3d_body(str(ROOT / "models/sam3d_body/model.ckpt"), device="cpu", mhr_path=str(ROOT / "models/sam3d_body/assets/mhr_model.pt"))
        _head = model.head_pose
    return _head


@torch.no_grad()
def forward(global_rot, body_pose, hand, scale, shape, chunk=256, want_verts=True):
    """Arrays (N, ...) -> verts (N, 18439, 3), keypoints (N, 70, 3), joints (N, 127, 3); camera-style axes, metres, root at origin."""
    h = head()
    V, K, J = [], [], []
    for i in range(0, len(global_rot), chunk):
        g, b, hd, sc, sh = (torch.as_tensor(np.asarray(x[i:i + chunk]), dtype=torch.float32) for x in (global_rot, body_pose, hand, scale, shape))
        v, k, j = h.mhr_forward(global_trans=torch.zeros_like(g), global_rot=g, body_pose_params=b, hand_pose_params=hd, scale_params=sc, shape_params=sh,
                                expr_params=torch.zeros(len(g), h.num_face_comps), return_keypoints=True, return_joint_coords=True)
        for arr, out in ((v, V), (k[:, :70], K), (j, J)):
            arr = arr.clone(); arr[..., [1, 2]] *= -1
            out.append(arr.numpy() if (out is not V or want_verts) else None)
    cat = lambda xs: np.concatenate(xs) if xs[0] is not None else None  # noqa: E731
    return cat(V), cat(K), cat(J)
