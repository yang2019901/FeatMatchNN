import matplotlib.patches
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


# given img1, points1 and img2, return the matched points2 in img2
class FeatMatchNN(nn.Module):
    """Feature Matching Neural Network
    Note: all input and output are torch.Tensor
    """

    def __init__(self):
        super(FeatMatchNN, self).__init__()
        c1, c2, c3, c4 = 3, 8, 16, 32
        self.conv_enc = nn.ModuleList(
            [
                nn.Conv2d(c1, c2, kernel_size=3, padding=1),
                nn.Conv2d(c2, c3, kernel_size=3, padding=1),
                nn.Conv2d(c3, c4, kernel_size=3, padding=1),
            ]
        )
        self.conv_dec = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    c4, c3, kernel_size=3, stride=2, padding=1, output_padding=1
                ),
                nn.ConvTranspose2d(
                    c3, c2, kernel_size=3, stride=2, padding=1, output_padding=1
                ),
                nn.ConvTranspose2d(
                    c2, 1, kernel_size=3, stride=2, padding=1, output_padding=1
                ),
            ]
        )

    def forward(self, x1, x2, pts1):
        """
        x1, x2: (B, 3, H, W), elements in [0, 1] instead of [0, 255]
        pts1: (B, L, 2)

        return
            conf: (B, L, H, W), confidence map of x2, elements in (-inf, inf)
            valid: (B, L), boolean tensor
        """
        B, L = pts1.shape[:2]
        H, W = x1.shape[-2:]
        # copy and reshape
        h1 = x1.clone().detach().view(B, 3, H, W)
        h2 = x2.clone().detach().view(B, 3, H, W)
        pts = pts1.clone().detach().view(B, L, 2)
        # forward h1, h2 and pts
        attns = []
        for i, layer in enumerate(self.conv_enc):
            h1 = F.leaky_relu(layer(h1))
            h2 = F.leaky_relu(layer(h2))  # (B, Ci, Hi, Wi)
            Ci, Hi, Wi = h1.shape[-3:]
            # get Q that is used to calculate attention, Q: (B, Ci, L)
            # Q[i, j, k] = h1[i, j, pts[i, k, 0], pts[i, k, 1]]
            Q = h1.view(B, Ci, Hi * Wi).gather(
                2,
                (pts[:, :, 0] * Wi + pts[:, :, 1]).unsqueeze(1).expand(-1, Ci, -1),
            )
            attn = F.softmax(
                torch.einsum("bchw,bcl->bhwl", h2, Q).view(B, L, -1), dim=-1
            ).view(B * L, 1, Hi, Wi)
            attns.append(attn)
            h1 = F.max_pool2d(h1, 2)
            h2 = F.max_pool2d(h2, 2)
            pts = pts // 2

        for i, layer in enumerate(self.conv_dec):
            attn = attns.pop()
            # h2: (B[*L], Ci, Hi, Wi), attn: (B*L, 1, Hi, Wi)
            h2 = layer(h2) * attn
            if i < len(self.conv_dec) - 1:
                h2 = F.leaky_relu(h2)
        # h2: (B*L, 1, H, W)
        conf = h2.view(B, L, H, W)
        valid = torch.max(conf.view(B, L, -1), dim=-1)[0] > 1
        return conf, valid

    def match(self, x1, x2, pts1):
        """
        x1, x2: ([B, ]3, H, W), elements in [0, 1] instead of [0, 255]
        pts1: ([B, ]L, 2)

        return
            pts2: (B, L, 2)
            valid: (B, L), boolean tensor
        """
        H, W = x1.shape[-2:]
        B = x1.shape[0] if len(x1.shape) == 4 else 1
        L = pts1.shape[-2]
        conf, valid = self.forward(
            x1.view(B, 3, H, W), x2.view(B, 3, H, W), pts1.view(B, L, 2)
        )
        pts2 = self.conf2pts(conf, valid, H, W)
        return pts2, valid

    def loss(self, x1, x2, pts1, pts2, valid2, cld2):
        """argmax the confidence map and calculate the distance
        ---
        x1, x2: (B, 3, H, W), elements in [0, 1] instead of [0, 255]
        pts1, pts2: (B, L, 2)
        valid2: (B, L)
        cld2: (B, H, W, 3)

        return
            loss: scalar
        """
        B, L = pts1.shape[:2]
        H, W = x1.shape[-2:]
        # predict confidence map and valid mask
        conf_pred, valid_pred = self.forward(x1, x2, pts1)
        pts_pred = self.conf2pts(conf_pred)  # (B, L, 2)
        # Note: argmax differentiable problem?
        cld = cld2.view(B, H * W, 3)
        valid = valid2.view(B, L, 1)
        pts = pts2.view(B, L, 2)
        pts2_3d_pred = valid_pred.view(B, L, 1) * cld.gather(
            1, (pts_pred[..., 0] * W + pts_pred[..., 1]).unsqueeze(-1).expand(-1, -1, 3)
        )  # (B, L, 3)
        pts2_3d = valid * cld.gather(
            1, (pts[..., 0] * W + pts[..., 1]).unsqueeze(-1).expand(-1, -1, 3)
        )  # (B, L, 3)
        return torch.mean(torch.norm(pts2_3d_pred - pts2_3d, dim=-1))

    def loss2(self, x1, x2, pts1, pts2, valid2, cld2):
        """softmax the confidence map and calculate weighted sum of distances
        ---
        x1, x2: (B, 3, H, W), elements in [0, 1] instead of [0, 255]
        pts1, pts2: (B, L, 2)
        valid2: (B, L)
        cld2: (B, H, W, 3)

        return
            loss: scalar
        """
        B, L = pts1.shape[:2]
        H, W = x1.shape[-2:]
        conf_pred, _ = self.forward(x1, x2, pts1)
        cld = cld2.view(B, H * W, 3)
        valid = valid2.view(B, L, 1)
        pts = pts2.view(B, L, 2)
        # pts2_3d: (B, L, 3)
        pts2_3d = valid * cld.gather(
            1, (pts[..., 0] * W + pts[..., 1]).unsqueeze(-1).expand(-1, -1, 3)
        )
        # dist: (B, L, H*W)
        dist = torch.norm(pts2_3d.view(B, L, 1, 3) - cld.view(B, 1, H * W, 3), dim=-1)
        weight = torch.softmax(conf_pred.view(B, L, -1), dim=-1)
        return torch.sum(weight * dist) / (B * L)

    def pts2conf(self, pts, valid, H, W):
        """generate confidence map from points and valid
        ---
        pts: (B, L, 2), valid: (B, L)

        return
            conf: (B, L, H, W)
        """
        B, L = valid.shape[-2:]
        conf = torch.zeros((B, L, H * W))
        idx = (pts[:, :, 0] * W + pts[:, :, 1]).unsqueeze(-1)
        valid = valid.unsqueeze(-1)
        conf.scatter_(dim=2, index=idx, src=valid)
        return conf

    def conf2pts(self, conf):
        """get the most confident points' indices
        ---
        conf: (B, L, H, W)

        return
            pts: (B, L, 2)
        """
        H, W = conf.shape[-2:]
        y = conf.flatten(start_dim=-2)
        indices = torch.argmax(y, dim=-1)
        pts = torch.stack([indices // W, indices % W], dim=-1)
        return pts


def preprocessing():
    imgs, clds = torch.load("carbinet.dat")
    imgs = imgs.permute(0, 3, 1, 2)  # (B, H, W, 3) -> (B, 3, H, W), range [0, 1]
    SP_match(imgs[0], imgs[1])


def confirm(x1, x2, m_pts1, m_pts2, um_pts1):
    """draw the matched points one by one and manually confirm them"""
    ratio = 4 / 3
    fig, axes = plt.subplots(1, 2, figsize=(2 * ratio * 4.5, 4.5))
    axes[0].imshow(x1.permute(1, 2, 0).cpu().numpy())
    axes[0].axis("off")
    axes[1].imshow(x2.permute(1, 2, 0).cpu().numpy())
    axes[1].axis("off")
    l1 = m_pts1.shape[0]
    l2 = um_pts1.shape[0]
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


def SP_match(x1, x2):
    """match two images using SuperPoint and LightGlue
    ---
    x1, x2: (3, H, W), elements in [0, 1] instead of [0, 255]

    return:
        m_pts1, m_pts2: matched points in x1 and x2
        um_pts1: unmatched points in x1
    """
    from LightGlue.lightglue import LightGlue, SuperPoint, viz2d
    from LightGlue.lightglue.utils import rbd, numpy_image_to_torch

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
    m_pts1, m_pts2 = pts1[matches[..., 0]].to(torch.int), pts2[matches[..., 1]].to(torch.int)
    um_pts1 = pts1[[i for i in range(len(pts1)) if i not in matches[..., 0]]][:30].to(torch.int)
    correct_m, correct_um = confirm(x1, x2, m_pts1, m_pts2, um_pts1)
    return m_pts1[correct_m], m_pts2[correct_m], um_pts1[correct_um]


def main():
    # Note: ensure that H and W is power of 2
    torch.manual_seed(0)
    H = W = 8
    L = 2
    B = 1
    model = FeatMatchNN()
    x1 = torch.rand(B, 3, H, H)
    x2 = torch.rand(B, 3, H, H)
    pts1 = torch.randint(0, H, (B, L, 2))
    pts2 = torch.randint(0, H, (B, L, 2))
    valid2 = torch.rand(B, L) > 0.5
    cld2 = torch.rand(B, H, H, 3)
    loss = model.loss2(x1, x2, pts1, pts2, valid2, cld2)
    print(loss)


# main()
preprocessing()
