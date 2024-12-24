import matplotlib.patches
import torch
import matplotlib.pyplot as plt
import matplotlib
import open3d as o3d
import os

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
def autocheck(m_pts1, m_pts2, um_pts1, cld1_glb, cld2_glb, k_tol=0.03):
    """generate boolean mask of matched points, using distance threshold
    ---
    m_pts1, m_pts2: (L1, 2), int
    um_pts1: (L2, 2), int
    cld1_glb, cld2_glb: (H, W, 3), float32

    return:
        matched: (L1, ), bool
        um_matched: (L2, ), bool
    """
    dist1 = torch.norm(cld1_glb, dim=-1)  # (H, W)
    dist2 = torch.norm(cld2_glb, dim=-1)  # (H, W)
    err1 = torch.norm(
        cld1_glb[m_pts1[:, 1], m_pts1[:, 0]] - cld2_glb[m_pts2[:, 1], m_pts2[:, 0]],
        dim=-1,
    )
    err1_tol = dist1[m_pts1[:, 1], m_pts1[:, 0]] + dist2[m_pts2[:, 1], m_pts2[:, 0]]
    matched = err1 < err1_tol * k_tol
    if um_pts1 is None:
        return matched, None
    err2 = torch.norm(
        cld1_glb[um_pts1[:, 1], um_pts1[:, 0]].view(-1, 1, 3) - cld2_glb.view(1, -1, 3),
        dim=-1,
    )  # (L2, H*W)
    err2_tol = dist1[um_pts1[:, 1], um_pts1[:, 0]].view(-1, 1) + dist2.view(1, -1)
    unmatched = torch.min(err2 / err2_tol, dim=-1)[0] > k_tol
    return matched, unmatched


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
    assert L1 + L2 >= L
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
    imgs = torch.tensor(imgs, dtype=torch.uint8)
    clds = torch.tensor(clds, dtype=torch.float32)
    X = (
        imgs.permute(0, 3, 1, 2) / 255.0
    )  # (N, H, W, 3) -> (N, 3, H, W), range: [0, 255] -> [0, 1]
    N, _, H, W = X.shape
    # extract features
    dev = torch.device("cuda:0")
    extractor = SuperPoint(max_num_keypoints=4 * L).to(dev)
    matcher = LightGlue(features="superpoint").eval().to(dev)
    feats = [extractor.extract(X[i].to(dev)) for i in range(N)]
    # match
    frames = []
    for i in range(1, N):
        feat1, cld1, pose1 = feats[0], clds[0], poses[0]
        feat2, cld2, pose2 = feats[i], clds[i], poses[i]
        matches12 = matcher({"image0": feat1, "image1": feat2})
        feat1, feat2, matches12 = rbd(feat1), rbd(feat2), rbd(matches12)
        pts1, pts2, matches = (
            feat1["keypoints"].to("cpu"),
            feat2["keypoints"].to("cpu"),
            matches12["matches"].to("cpu"),
        )
        m_pts1, m_pts2 = pts1[matches[..., 0]].to(torch.int), pts2[matches[..., 1]].to(
            torch.int
        )
        um_pts1 = pts1[[i for i in range(len(pts1)) if i not in matches[..., 0]]].to(
            torch.int
        )
        cld1_glb, cld2_glb = transform(cld1, pose1), transform(cld2, pose2)
        correct_m, correct_um = autocheck(m_pts1, m_pts2, um_pts1, cld1_glb, cld2_glb)
        pts1, pts2, valid2 = sample(
            m_pts1[correct_m], m_pts2[correct_m], um_pts1[correct_um], L
        )
        frames.append((X[0], X[i], pts1, pts2, valid2, clds[i]))
    print(f"{N-1} pairs of images are processed.")
    return frames


def viz_frame(frame):
    x1, x2, pts1, pts2, valid2, cld2 = frame
    viz2d.plot_images((x1, x2))
    viz2d.plot_matches(pts1[valid2], pts2[valid2], color="lime", lw=0.4)
    viz2d.plot_keypoints([pts1[~valid2], ], ps=6)
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