import matplotlib.patches
import torch
import matplotlib.pyplot as plt
import matplotlib
import open3d as o3d
import os
import numpy as np

from LightGlue.lightglue import LightGlue, SuperPoint, viz2d
from LightGlue.lightglue.utils import rbd, numpy_image_to_torch


def quat2mat(quaternion):
    q = torch.tensor(quaternion[:4], dtype=torch.float)
    nq = torch.dot(q, q)
    if nq < torch.finfo(torch.float).eps:
        return torch.eye(3, dtype=torch.float)
    q *= torch.sqrt(2.0 / nq)
    q = torch.ger(q, q)
    return torch.tensor(
        (
            (1.0 - q[1, 1] - q[2, 2], q[0, 1] - q[2, 3], q[0, 2] + q[1, 3]),
            (q[0, 1] + q[2, 3], 1.0 - q[0, 0] - q[2, 2], q[1, 2] - q[0, 3]),
            (q[0, 2] - q[1, 3], q[1, 2] + q[0, 3], 1.0 - q[0, 0] - q[1, 1]),
        ),
        dtype=torch.float,
    )


def transform(cld, pose):
    """transform the point cloud using the pose
    ---
    cld: (..., 3)
    pose: [[x, y, z], [qx, qy, qz, qw]]

    return:
        cld_glb: (..., 3)
    """
    t = torch.tensor(pose[0])
    R = quat2mat(pose[1])
    cld_glb = cld @ R.T + t
    return cld_glb


def confirm(x1, x2, m_pts1, m_pts2, um_pts1=None):
    """draw the matched points one by one and manually confirm them"""
    ratio = 4 / 3
    fig, axes = plt.subplots(1, 2, figsize=(2 * ratio * 4.5, 4.5))
    axes[0].imshow(x1.permute(1, 2, 0).cpu().numpy())
    axes[0].axis("off")
    axes[1].imshow(x2.permute(1, 2, 0).cpu().numpy())
    axes[1].axis("off")
    l1 = m_pts1.shape[0]
    l2 = um_pts1.shape[0] if um_pts1 is not None else 0
    # draw matched points
    axes[0].scatter(m_pts1[:, 0], m_pts1[:, 1], c="g", s=4)
    axes[1].scatter(m_pts2[:, 0], m_pts2[:, 1], c="g", s=4)
    for i in range(l1):
        conn = matplotlib.patches.ConnectionPatch(
            xyA=m_pts1[i],
            xyB=m_pts2[i],
            coordsA=axes[0].transData,
            coordsB=axes[1].transData,
            color="g",
        )
        fig.add_artist(conn)
    # draw unmatched points
    axes[0].scatter(um_pts1[:, 0], um_pts1[:, 1], c="b", s=4)
    # draw indicator
    c1 = plt.Circle(m_pts1[0], 3, color="r")
    c2 = plt.Circle(m_pts2[0], 3, color="r")
    conn = matplotlib.patches.ConnectionPatch(
        xyA=m_pts1[0],
        xyB=m_pts2[0],
        coordsA=axes[0].transData,
        coordsB=axes[1].transData,
        color="r",
    )
    axes[0].add_artist(c1)
    axes[1].add_artist(c2)
    fig.add_artist(conn)
    # indicator and data
    idx = 0
    correct = torch.zeros(l1 + l2, dtype=torch.bool)  # whether lightglue is correct
    fig.suptitle(f"idx: {idx} / ({l1} + {l2})")
    fig.tight_layout(pad=0)

    def on_key(event):
        """
        n/m: next/previous point pair;
        j/k: accept/reject the current point pair
        """
        nonlocal idx, c1, c2, conn, correct
        if event.key == "q":
            plt.close()
            return
        correct[idx] = (
            1 if event.key == "j" else (0 if event.key == "k" else correct[idx])
        )
        # change idx
        if event.key == "n" or event.key == "j" or event.key == "k":
            idx += 1
        elif event.key == "m" and idx > 0:
            idx -= 1
        if idx >= l1 + l2:
            plt.close()
            return
        # update drawing
        fig.suptitle(f"idx: {idx} / ({l1} + {l2})")
        c1.center = m_pts1[idx] if idx < l1 else um_pts1[idx - l1]
        c2.set_visible(idx < l1)
        conn.set_visible(idx < l1)
        if idx < l1:
            c2.center = m_pts2[idx]
            conn.xy1 = m_pts1[idx]
            conn.xy2 = m_pts2[idx]
        fig.canvas.draw()

    fig.canvas.mpl_connect(
        "key_press_event",
        on_key,
    )
    plt.show()
    print("lightglue correct: ", correct)
    return correct[:l1], correct[l1:]


# Note: distance noise gets larger when obj is further, here we use a ratio `k_tol`: (p1 - p2) < (norm(p1) + norm(p2)) * k_tol
def autocheck(pts1, cld1_glb, cld2_glb, k_tol=0.01):
    """
    pts1: (L, 2), int
    cld1_glb, cld2_glb: (H, W, 3), float32

    return:
        pts2: (L, 2), int
        valid2: (L,), bool
    """
    H, W = cld1_glb.shape[:2]
    L = pts1.shape[0]
    dist1 = torch.norm(cld1_glb, dim=-1)  # (H, W)
    dist2 = torch.norm(cld2_glb, dim=-1)  # (H, W)
    pts1_3d = cld1_glb[pts1[:, 0], pts1[:, 1]]  # (L, 3)
    err = torch.norm(
        cld2_glb.view(1, H * W, 3) - pts1_3d.view(L, 1, 3) + 1e-6, dim=-1
    )  # (L, H*W)
    k_err = err / (
        dist1[pts1[:, 0], pts1[:, 1]].view(L, 1) + dist2.view(1, -1) + 1e-6
    )  # (L, H*W)
    min_k_err, idx = torch.min(k_err, dim=-1)
    pts2 = torch.stack([idx // W, idx % W], dim=-1)
    valid2 = min_k_err < k_tol

    i = 0
    _, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(cld1_glb[..., 2])
    # axes[1].imshow(cld2_glb[..., 2])
    axes[1].imshow(k_err[i].view(H, W).cpu().numpy())
    axes[0].scatter(pts1[i, 1], pts1[i, 0], c="r", s=4)
    axes[1].scatter(pts2[i, 1], pts2[i, 0], c="r", s=4)
    plt.show()

    return pts2, valid2


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
    valid2 = torch.cat(
        [torch.ones(l1, dtype=torch.bool), torch.zeros(l2, dtype=torch.bool)]
    )
    print("match, unmatch: ", (l1, l2))
    return pts1, pts2, valid2


def MakeDataset(imgs, clds, poses, L):
    """assume imgs/clds[0] is object image, imgs/clds[1:] are scene images.
    extract `L` point pairs from each scene image and the object image.

    return:
        frames: list of N-1 frame. frame format: (x1, x2, pts1, pts2, valid2, cld2)
    """
    imgs = imgs.to(torch.uint8)
    clds = clds.to(torch.float32)
    X = (
        imgs.permute(0, 3, 1, 2) / 255.0
    )  # (N, H, W, 3) -> (N, 3, H, W), range: [0, 255] -> [0, 1]
    N, _, H, W = X.shape
    # extract features
    dev = torch.device("cuda:0")
    extractor = SuperPoint(max_num_keypoints=4 * L).to(dev)
    feats = [extractor.extract(X[i].to(dev)) for i in range(N)]
    # match
    frames = []
    for i in range(1, N):
        feat1, cld1, pose1 = feats[0], clds[0], poses[0]
        feat2, cld2, pose2 = feats[i], clds[i], poses[i]
        pts1 = feat1["keypoints"].to("cpu", dtype=int).flip([-1]).view(-1, 2)
        cld1_glb, cld2_glb = transform(cld1, pose1), transform(cld2, pose2)

        # use open3d registration to fine-tune the pose
        pcd1, pcd2 = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
        pcd1.points, pcd2.points = (
            o3d.utility.Vector3dVector(cld1_glb.view(-1, 3).cpu().numpy()),
            o3d.utility.Vector3dVector(cld2_glb.view(-1, 3).cpu().numpy()),
        )
        result = o3d.pipelines.registration.registration_icp(
            pcd1, pcd2, 0.1, np.eye(4, dtype=np.float32), o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        pose = result.transformation
        pos, rot = pose[0:3, 3], pose[0:3, 0:3]
        cld1_glb = cld1_glb @ rot.T + pos
        pcd1.points = o3d.utility.Vector3dVector(cld1_glb.view(-1, 3).cpu().numpy())

        pts2, valid2 = autocheck(pts1, cld1_glb, cld2_glb)
        pts1, pts2, valid2 = sample(pts1[valid2], pts2[valid2], pts1[~valid2], L)
        frames.append((X[0], X[i], pts1, pts2, valid2, clds[i]))
        viz_frame(frames[-1])
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
