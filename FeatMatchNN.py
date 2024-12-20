import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
        x1, x2: (3, H, W), elements in [0, 1] instead of [0, 255]
        pts1: (B, 2)
        return
            conf: (B, H, W), confidence map of x2, elements in (-inf, inf)
            valid: (B, ), boolean tensor
        """
        h1, h2 = x1.clone().detach(), x2.clone().detach()
        pts = pts1.clone().detach()
        B = pts.shape[0]
        attns = []
        for i, layer in enumerate(self.conv_enc):
            Hi, Wi = h1.shape[1:]
            h1 = F.relu(layer(h1))
            h2 = F.relu(layer(h2))  # (Ci, Hi, Wi)
            Q = h1[:, pts[:, 0], pts[:, 1]]  # (Ci, B)
            attn = F.softmax(
                torch.einsum("chw,cb->bhw", h2, Q).view(B, -1), dim=1
            ).view(B, 1, Hi, Wi)
            attns.append(attn)
            h1 = F.max_pool2d(h1, 2)
            h2 = F.max_pool2d(h2, 2)
            pts = pts // 2

        for i, layer in enumerate(self.conv_dec):
            attn = attns.pop()
            # h2: ([B, ]Ci, Hi, Wi), attn: (B, 1, Hi, Wi)
            h2 = layer(h2) * attn
            if i < len(self.conv_dec) - 1:
                h2 = F.relu(h2)
        # h2: (B, 1, H, W)
        conf = h2.squeeze(1)
        valid = torch.max(conf.view(B, -1), dim=1)[0] > 1
        return conf, valid

    def match(self, x1, x2, pts1):
        """
        x1, x2: (3, H, W), elements in [0, 1] instead of [0, 255]
        pts1: (B, 2)
        return
            pts2: (B, 2)
            valid: (B, ), boolean tensor
        """
        conf, valid = self.forward(x1, x2, pts1)
        B, H, W = conf.shape
        y = conf.view(B, -1)
        indices = torch.argmax(y, dim=1)
        pts2 = torch.stack([indices // W, indices % W], dim=1)
        return pts2, valid

    def loss(self, x1, x2, pts1, pts2, valid2, cld2):
        """argmax the confidence map and calculate the distance
        x1, x2: (3, H, W), elements in [0, 1] instead of [0, 255]
        pts1, pts2: (B, 2)
        valid2: (B, )
        cld2: (H, W, 3)
        return
            loss: scalar
        """
        B = pts1.shape[0]
        conf_pred, valid_pred = self.forward(x1, x2, pts1)
        # Note: argmax differentiable problem?
        pts2_pred = self.conf2pts(conf_pred)
        pts2_3d_pred = cld2[pts2_pred[:, 0], pts2_pred[:, 1]] * valid_pred.view(B, 1)
        pts2_3d = cld2[pts2[:, 0], pts2[:, 1]] * valid2.view(B, 1)
        return torch.mean(torch.norm(pts2_3d_pred - pts2_3d, dim=1))

    def loss2(self, x1, x2, pts1, pts2, valid2, cld2):
        """softmax the confidence map and calculate weighted sum of distances
        x1, x2: (3, H, W), elements in [0, 1] instead of [0, 255]
        pts1, pts2: (B, 2)
        valid2: (B, )
        cld2: (H, W, 3)
        return
            loss: scalar
        """
        B = pts1.shape[0]
        pts2_3d = cld2[pts2[:, 0], pts2[:, 1]] * valid2.view(B, 1)
        conf_pred, _ = self.forward(x1, x2, pts1)
        # dist: (B, H*W)
        dist = torch.norm(pts2_3d.view(B, 1, 3) - cld2.view(1, -1, 3), dim=-1)
        return torch.sum(torch.softmax(conf_pred.view(B, -1), dim=1) * dist) / B

    def pts2conf(self, pts, valid, H, W):
        """generate confidence map from points and valid
        pts: (B, 2), valid: (B, )
        return
            conf: (B, H, W)
        """
        B = pts.shape[0]
        conf = np.zeros((B, H, W))
        for i in range(B):
            conf[i, pts[i, 0], pts[i, 1]] = valid[i]
        return conf

    def conf2pts(self, conf):
        """get the most confident points' indices
        conf: (B, H, W)
        return
            pts: (B, 2)
        """
        B, H, W = conf.shape
        y = conf.view(B, -1)
        indices = torch.argmax(y, dim=1)
        pts = torch.stack([indices // W, indices % W], dim=1)
        return pts


def main():
    # Note: ensure that L is power of 2
    L = 16
    B = 10
    model = FeatMatchNN()
    x1 = torch.rand(3, L, L)
    x2 = torch.rand(3, L, L)
    pts1 = torch.randint(0, L, (B, 2))
    pts2 = torch.randint(0, L, (B, 2))
    valid2 = torch.rand(B) > 0.5
    cld2 = torch.rand(L, L, 3)
    loss = model.loss2(x1, x2, pts1, pts2, valid2, cld2)
    print(loss)


main()
