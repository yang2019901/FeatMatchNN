import matplotlib.patches
import torch
import matplotlib.pyplot as plt
import matplotlib
import open3d as o3d
import os
import numpy as np
from numpy.linalg import inv

from LightGlue.lightglue import LightGlue, SuperPoint, viz2d
from LightGlue.lightglue.utils import rbd, numpy_image_to_torch

from scipy.spatial.transform import Rotation


## unity camera intrinsics
fov = 60 * np.pi / 180
h, w = 256, 256
fx, fy = w / (2 * np.tan(fov / 2)), h / (2 * np.tan(fov / 2))
cx, cy = w / 2, h / 2
cam_in = torch.Tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).to(torch.float32)


def transform(cld, mat_pose):
    """transform the point cloud using the pose"""
    t = mat_pose[:3, 3]
    R = mat_pose[:3, :3]
    return cld @ R.T + t


def pose2mat(pose):
    """convert pose to transformation matrix
    pose: [[x, y, z], [qx, qy, qz, qw]]

    return:
        M: (4, 4), float32
    """
    t = torch.tensor(pose[0])
    R = Rotation.from_quat(pose[1]).as_matrix()
    M = np.eye(4, dtype=np.float32)
    M[:3, :3], M[:3, 3] = R, t
    return M


def annotate(pts1, cld1, cld2, cam_in, k_tol=0.01):
    """auto-annotate the matched points with imaging principle
    pts1: (L, 2), int
    cld1: (H, W, 3), float32
    cld2: (H, W, 3), float32
    cam_in: (3, 3), float32
    k_tol: float, tolerance ratio of the distance between the projected point and the nearest point

    return:
        pts2: (L, 2), int
        valid2: (L,), bool
    """
    H, W = cld1.shape[:2]
    L = pts1.shape[0]
    ## project 3d points to 2d
    pts1_3d = cld1[pts1[:, 0], pts1[:, 1]].T  # (3, L)
    uv2 = (cam_in @ (pts1_3d / pts1_3d[2, :]))[:2]  # (2, L)
    pts2_f = uv2.T.flip([-1])  # (L, 2)

    ## floor/ceil x/y and get the nearest among 4 points
    pts2_fx_fy = pts2_f.floor().to(torch.int)
    pts2_cy_cy = pts2_f.ceil().to(torch.int)
    pts2_fx_cy = torch.stack([pts2_fx_fy[:, 0], pts2_cy_cy[:, 1]], dim=-1)
    pts2_cx_fy = torch.stack([pts2_cy_cy[:, 0], pts2_fx_fy[:, 1]], dim=-1)
    ## four corners of the nearest 4 points
    pts2_cnrs = torch.stack([pts2_fx_fy, pts2_cy_cy, pts2_fx_cy, pts2_cx_fy], dim=1)  # (L, 4, 2)
    pts2_cnrs.clamp_(
        torch.zeros(1, 1, 2, dtype=torch.int), max=torch.tensor([[[H - 1, W - 1]]], dtype=torch.int)
    )  # (L, 4, 2)
    pts2_cnrs = W * pts2_cnrs[:, :, 0] + pts2_cnrs[:, :, 1]  # (L, 4)

    pts2 = torch.zeros(L, dtype=torch.int)
    valid2 = np.zeros(L, dtype=bool)
    cld2 = cld2.view(-1, 3)
    for i in range(L):
        p0 = pts1_3d[:, i]
        p = cld2[pts2_cnrs[i], :]
        dist = torch.norm(p - p0, dim=-1)
        min_dist, min_idx = torch.min(dist, dim=0)
        valid2[i] = min_dist < k_tol * torch.norm(p0)
        pts2[i] = pts2_cnrs[i, min_idx]
    pts2 = torch.stack([pts2 // W, pts2 % W], dim=-1)

    return pts2, valid2


def tensor2pcd(cld, rgb):
    """cld: (H, W, 3), float32"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cld.reshape(-1, 3))
    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb.reshape(-1, 3))
    return pcd


def sample(m_pts1, m_pts2, um_pts1, L, k_valid=0.8):
    """sample L point pairs from matched points and unmatched points
    ---
    m_pts1, m_pts2: (L1, 2), int
    um_pts1: (L2, 2), int
    L: int

    return:
        pts1, pts2: (L, 2), int
        valid2: (L, ), bool
    """
    L1, L2 = m_pts1.shape[0], um_pts1.shape[0]
    assert L1 + L2 >= L, f"insufficient keypoints: {L1} + {L2} < {L}"
    l1 = max(min(int(k_valid * L), L1), L - L2)
    l2 = L - l1
    idx_m = torch.randperm(L1)[:l1]
    idx_um = torch.randperm(L2)[:l2]
    pts1 = torch.cat([m_pts1[idx_m], um_pts1[idx_um]], dim=0)
    pts2 = torch.cat([m_pts2[idx_m], torch.zeros(l2, 2, dtype=torch.int)], dim=0)
    valid2 = torch.cat([torch.ones(l1, dtype=torch.bool), torch.zeros(l2, dtype=torch.bool)])
    print("match, unmatch: ", (l1, l2))
    return pts1, pts2, valid2


def MakeDataset(imgs, clds, poses, L):
    """assume imgs/clds[0] is object image, imgs/clds[1:] are scene images.
    extract `L` point pairs from each scene image and the object image.

    return:
        frames: list of N-1 frame. frame format: (x1, x2, pts1, pts2, valid2, cld2)
    """
    global cam_in
    imgs = imgs.to(torch.uint8)
    clds = clds.to(torch.float32)
    X = imgs.permute(0, 3, 1, 2) / 255.0  # (N, H, W, 3) -> (N, 3, H, W), range: [0, 255] -> [0, 1]
    N, _, H, W = X.shape
    ## extract features
    dev = torch.device("cuda:0")
    extractor = SuperPoint(max_num_keypoints=4 * L).to(dev)
    feats = [extractor.extract(X[i].to(dev)) for i in range(N)]
    ## match
    frames = []
    for i in range(1, N):
        feat1, cld1, pose1 = feats[0], clds[0], poses[0]
        feat2, cld2, pose2 = feats[i], clds[i], poses[i]
        pts1 = feat1["keypoints"].to("cpu", dtype=int).flip([-1]).view(-1, 2)
        M1, M2 = pose2mat(pose1), pose2mat(pose2)
        cld1 = transform(cld1, inv(M2) @ M1)
        pts2, valid2 = annotate(pts1, cld1, cld2, cam_in)
        viz_frame((X[0], X[i], pts1, pts2, valid2, cld2))
        pts1, pts2, valid2 = sample(pts1[valid2], pts2[valid2], pts1[~valid2], L)
        frames.append((X[0], X[i], pts1, pts2, valid2, clds[i]))
    print(f"{N-1} pairs of images are processed.")
    return frames


def viz_frame(frame):
    x1, x2, pts1, pts2, valid2, cld2 = frame
    uv1, uv2 = pts1.flip([-1]), pts2.flip([-1])
    viz2d.plot_images((x1, x2))
    viz2d.plot_matches(uv1[valid2], uv2[valid2], color="lime", lw=0.4)
    viz2d.plot_keypoints([uv1[~valid2]], ps=6)
    plt.show()


if __name__ == "__main__":
    raw_dir = os.path.join(os.path.dirname(__file__), "raw")
    dataset_dir = os.path.join(os.path.dirname(__file__), "dataset")
    for f in os.listdir(raw_dir):
        print(f"processing {f}")
        imgs, clds, poses = torch.load(f"{raw_dir}/{f}", weights_only=True)
        li = MakeDataset(imgs, clds, poses, L=100)
        torch.save(li, f"{dataset_dir}/{f}")
    viz_frame(li[-1])
