import matplotlib.patches
import torch
import matplotlib.pyplot as plt
import matplotlib
import open3d as o3d

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


def autocheck(m_pts1, m_pts2, um_pts1, cld1_glb, cld2_glb, dist_thresh=0.02):
    """generate boolean mask of matched points, using distance threshold
    ---
    m_pts1, m_pts2: (L1, 2), int
    um_pts1: (L2, 2), int
    cld1_glb, cld2_glb: (H, W, 3), float32

    return:
        matched: (L1, ), bool
        um_matched: (L2, ), bool
    """
    dist1 = torch.norm(
        cld1_glb[m_pts1[:, 1], m_pts1[:, 0]] - cld2_glb[m_pts2[:, 1], m_pts2[:, 0]],
        dim=-1,
    )
    matched = dist1 < dist_thresh
    if um_pts1 is None:
        return matched, None
    dist2 = torch.norm(
        cld1_glb[um_pts1[:, 1], um_pts1[:, 0]].view(-1, 1, 3) - cld2_glb.view(1, -1, 3),
        dim=-1,
    )  # (L2, H*W)
    dist2 = torch.min(dist2, dim=-1)[0]
    um_matched = dist2 > dist_thresh
    return matched, um_matched


def SP_match(x1, x2, cld1, cld2, pose1, pose2):
    """match two images using SuperPoint and LightGlue
    ---
    x1, x2: (3, H, W), elements in [0, 1] instead of [0, 255]
    cld1, cld2: (H, W, 3), float32
    pose1, pose2: [[x, y, z], [qx, qy, qz, qw]]; cam1 and cam2's pose in the reference frame

    return:
        m_pts1, m_pts2: matched points in x1 and x2
        um_pts1: unmatched points in x1
    """
    dev = torch.device("cuda:0")
    extractor = SuperPoint(max_num_keypoints=256).to(dev)
    matcher = LightGlue(features="superpoint").eval().to(dev)
    feats1 = extractor.extract(x1.to(dev))
    feats2 = extractor.extract(x2.to(dev))
    matches12 = matcher({"image0": feats1, "image1": feats2})
    feats1, feats2, matches12 = rbd(feats1), rbd(feats2), rbd(matches12)
    pts1, pts2, matches = (
        feats1["keypoints"].to("cpu"),
        feats2["keypoints"].to("cpu"),
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
    return m_pts1[correct_m], m_pts2[correct_m], um_pts1[correct_um]


def main():
    imgs, clds, poses = torch.load("carbinet.dat")
    imgs = torch.tensor(imgs, dtype=torch.uint8)
    clds = torch.tensor(clds, dtype=torch.float32)
    X = (
        imgs.permute(0, 3, 1, 2) / 255
    )  # (B, H, W, 3) -> (B, 3, H, W), range: [0, 255] -> [0, 1]

    SP_match(X[0], X[1], clds[0], clds[1], poses[0], poses[1])


if __name__ == "__main__":
    main()
