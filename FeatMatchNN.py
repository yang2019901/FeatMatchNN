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


def confirm(x0, x1, m_pts0, m_pts1):
    """draw the matched points one by one and manually confirm them"""
    ratio = 4 / 3
    fig, axes = plt.subplots(1, 2, figsize=(2 * ratio * 4.5, 4.5))
    axes[0].imshow(x0.permute(1, 2, 0).cpu().numpy())
    axes[0].axis("off")
    axes[1].imshow(x1.permute(1, 2, 0).cpu().numpy())
    axes[1].axis("off")
    l = m_pts0.shape[0]
    c0 = plt.Circle((m_pts0[0, 0], m_pts0[0, 1]), 2, color="r")
    c1 = plt.Circle((m_pts1[0, 0], m_pts1[0, 1]), 2, color="r")
    conn = matplotlib.patches.ConnectionPatch(
        xyA=(m_pts0[0, 0], m_pts0[0, 1]),
        xyB=(m_pts1[0, 0], m_pts1[0, 1]),
        coordsA=axes[0].transData,
        coordsB=axes[1].transData,
        color="r",
    )
    axes[0].add_artist(c0)
    axes[1].add_artist(c1)
    fig.add_artist(conn)
    idx = 0
    valid = torch.zeros(l, dtype=torch.bool)
    fig.suptitle(f"idx: {idx} / {l}")
    fig.tight_layout(pad=0)

    def on_key(event):
        nonlocal idx, c0, c1, conn, valid
        # print(f"key pressed: {event.key}\tidx: {idx} / {l}")
        if event.key == "q":
            plt.close()
            return
        # n, m to change to next or previous point pair
        # j, k to accept or reject the current point pair
        valid[idx] = 1 if event.key == "j" else (0 if event.key == "k" else valid[idx])
        # change idx
        if event.key == "n" or event.key == "j" or event.key == "k":
            idx += 1
        elif event.key == "m" and idx > 0:
            idx -= 1
        if idx >= l:
            plt.close()
            return
        fig.suptitle(f"idx: {idx} / {l}")
        c0.center = (m_pts0[idx, 0], m_pts0[idx, 1])
        c1.center = (m_pts1[idx, 0], m_pts1[idx, 1])
        conn.xy1 = (m_pts0[idx, 0], m_pts0[idx, 1])
        conn.xy2 = (m_pts1[idx, 0], m_pts1[idx, 1])
        fig.canvas.draw()

    fig.canvas.mpl_connect(
        "key_press_event",
        on_key,
    )
    plt.show()
    print("confirmed valid: ", valid)
    return m_pts0[valid], m_pts1[valid]


def SP_match(x0, x1):
    """match two images using SuperPoint and LightGlue
    ---
    x0, x1: (3, H, W), elements in [0, 1] instead of [0, 255]
    """
    from LightGlue.lightglue import LightGlue, SuperPoint, viz2d
    from LightGlue.lightglue.utils import rbd, numpy_image_to_torch

    dev = torch.device("cuda:0")
    extractor = SuperPoint(max_num_keypoints=256).to(dev)
    matcher = LightGlue(features="superpoint").eval().to(dev)
    feats0 = extractor.extract(x0.to(dev))
    feats1 = extractor.extract(x1.to(dev))
    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = rbd(feats0), rbd(feats1), rbd(matches01)
    pts0, pts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_pts0, m_pts1 = pts0[matches[..., 0]].to("cpu"), pts1[matches[..., 1]].to("cpu")
    m_pts0, m_pts1 = confirm(x0, x1, m_pts0, m_pts1)


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
