import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# given img1, points1 and img2, return the matched points2 in img2
class FeatMatchNN(nn.Module):
    def __init__(self):
        super(FeatMatchNN, self).__init__()
        c1, c2, c3, c4 = 3, 8, 16, 32
        self.bn = (nn.BatchNorm2d(c1),)
        self.conv_enc = nn.ModuleList(
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.Conv2d(c3, c4, kernel_size=3, padding=1),
        )
        self.conv_dec = nn.ModuleList(
            nn.ConvTranspose2d(
                c4, c3, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ConvTranspose2d(
                c3, c2, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ConvTranspose2d(
                c2, 1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
        )

    def forward(self, x1, x2, pts1):
        """
        x1, x2: (3, H, W), elements in [0, 1] instead of [0, 255]
        pts1: (B, 2)
        return
            conf: (B, H, W), valid: (B, )
        """
        attns = []
        h1 = self.bn(torch.tensor(x1))
        h2 = self.bn(torch.tensor(x2))
        pts = torch.tensor(pts1, dtype=torch.int16)
        B = pts.shape[0]
        for i, layer in enumerate(self.conv_enc):
            Hi, Wi = h1.shape[1:]
            h1 = F.relu(layer(h1))
            h2 = F.relu(layer(h2))  # (Ci, Hi, Wi)
            Q = h1[:, pts[:, 0], pts[:, 1]]  # (Ci, B)
            attn = F.softmax(torch.matmul(Q.T, h2).view(B, -1), dim=1).view(
                B, 1, Hi, Wi
            )
            attns.append(attn)
            h1 = F.max_pool2d(h1, 2)
            h2 = F.max_pool2d(h2, 2)
            pts = pts // 2

        for layer in self.conv_dec:
            attn = attns.pop()
            # h2: ([B, ]Ci, Hi, Wi), attn: (B, 1, Hi, Wi)
            h2 = F.relu(layer(h2) * attn)
        # h2: (B, 1, H, W)
        conf = h2.squeeze(1)
        valid = torch.max(conf, dim=1)[0] > 1
        return conf, valid

    def match(self, x1, x2, pts1):
        """
        x1, x2: (3, H, W), elements in [0, 1] instead of [0, 255]
        pts1: (B, 2)
        return
            pts2: (B, 2), valid: (B, )
        """
        conf, valid = self.forward(x1, x2, pts1)
        B, H, W = conf.shape
        y = conf.view(B, -1)
        indices = torch.argmax(y, dim=1)
        pts2 = torch.stack([indices // W, indices % W], dim=1)
        return pts2, valid

    def loss(self, x1, x2, pts1, pts2, valid2, cld1, cld2):
        """
        x1, x2: (3, H, W), elements in [0, 1] instead of [0, 255]
        pts1, pts2: (B, 2)
        valid2: (B, )
        cld1, cld2: (H, W, 3), Note: cld1 and cld2 are in same coordinate system
        return
            loss: scalar
        """
        B = pts1.shape
        conf1, valid1 = self.forward(x1, x2, pts1)
        pts1_3d = cld1[pts1[:, 0], pts1[:, 1]] * valid1
        pts2_3d = cld2[pts2[:, 0], pts2[:, 1]] * valid2
        return torch.mean(torch.norm(pts1_3d - pts2_3d, dim=1))

    def pts2conf(pts, valid, H, W):
        """
        pts: (B, 2), valid: (B, )
        return
            conf: (B, H, W)
        """
        B = pts.shape[0]
        conf = np.zeros((B, H, W))
        for i in range(B):
            conf[i, pts[i, 0], pts[i, 1]] = valid[i]
        return conf
