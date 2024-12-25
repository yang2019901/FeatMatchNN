import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchinfo import summary
import os


# given img1, points1 and img2, return the matched points2 in img2
class FeatMatchNN(nn.Module):
    """Feature Matching Neural Network
    Note: all input and output are torch.Tensor
    """

    def __init__(self):
        super(FeatMatchNN, self).__init__()
        c1, c2, c3, c4, c5 = 3, 8, 16, 32, 64
        # Note whether receptive field is large enough. See https://blog.csdn.net/Rolandxxx/article/details/127270974
        self.conv_enc = nn.ModuleList(
            [
                nn.Conv2d(c1, c2, kernel_size=3, padding=1),
                nn.Conv2d(c2, c3, kernel_size=5, padding=2),
                nn.Conv2d(c3, c4, kernel_size=7, padding=3),
                nn.Conv2d(c4, c5, kernel_size=9, padding=4),
            ]
        )
        self.conv_dec = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    c5, c4, kernel_size=3, stride=2, padding=1, output_padding=1
                ),
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
        pts = pts1.clone().detach().to(torch.int64).view(B, L, 2)
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
                torch.einsum("bchw,bcl->blhw", h2, Q).view(B, L, -1), dim=-1
            ).view(B * L, 1, Hi, Wi)
            attns.append(attn)
            h1 = F.max_pool2d(h1, 2)
            h2 = F.max_pool2d(h2, 2)
            # scale indices
            Hi_, Wi_ = h1.shape[-2:]
            pts[..., 0] = pts[..., 0] * Hi_ // Hi
            pts[..., 1] = pts[..., 1] * Wi_ // Wi

        h2 = h2.repeat((L, 1, 1, 1))
        for i, layer in enumerate(self.conv_dec):
            attn = attns.pop()
            # h2: (B*L, Ci, Hi, Wi), attn: (B*L, 1, Hi, Wi)
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

    def p2p_loss(self, x1, x2, pts1, pts2, valid2, cld2):
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

    def p2map_loss(self, x1, x2, pts1, pts2, valid2, cld2):
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


def viz_frame(frame):
    from DataProcess import viz_frame

    viz_frame(frame)


class FeatMatchDataset(torch.utils.data.Dataset):
    def __init__(self, frames, H=None, W=None, dev=None):
        self.x1, self.x2, self.pts1, self.pts2, self.valid2, self.cld2 = zip(*frames)
        self.x1 = torch.stack(self.x1).to(dev)
        self.x2 = torch.stack(self.x2).to(dev)
        self.pts1 = torch.stack(self.pts1).to(dev)
        self.pts2 = torch.stack(self.pts2).to(dev)
        self.valid2 = torch.stack(self.valid2).to(dev)
        self.cld2 = torch.stack(self.cld2).to(dev)
        self.H = frames[0][0].shape[-2] if H is None else H
        self.W = frames[0][0].shape[-1] if W is None else W
        self.transforms = v2.Compose(
            [
                v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                v2.GaussianBlur(3),
                v2.GaussianNoise(mean=0, sigma=0.03),
            ]
        )

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, idx):
        x1, x2, pts1, pts2, valid2, cld2 = self.x1[idx], self.x2[idx], self.pts1[idx], self.pts2[idx], self.valid2[idx], self.cld2[idx]
        H0, W0 = x1.shape[-2:]
        x1, x2, cld2 = v2.functional.resize(
            torch.stack([x1, x2, cld2.permute(2, 0, 1)]), (self.H, self.W)
        )
        cld2 = cld2.permute(1, 2, 0)
        # Note: the same transform for x1 and x2
        x1, x2 = self.transforms(torch.stack([x1, x2]))
        if self.H != H0:
            pts1[..., 0] = pts1[..., 0] * self.H // H0
            pts2[..., 0] = pts2[..., 0] * self.H // H0
        if self.W != W0:
            pts1[..., 1] = pts1[..., 1] * self.W // W0
            pts2[..., 1] = pts2[..., 1] * self.W // W0
        return x1, x2, pts1, pts2, valid2, cld2


def main():
    dev = torch.device("cuda")
    # load data
    dataset_dir = os.path.join(os.path.dirname(__file__), "dataset")
    frames = []
    for f in os.listdir(dataset_dir):
        frames += torch.load(f"{dataset_dir}/{f}", weights_only=True)
        print(f"{f} loaded.")
    # create dataset
    dataset = FeatMatchDataset(frames, H=128, W=128, dev=dev)
    # create dataloader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    # create model
    model = FeatMatchNN().to(dev)
    # train
    optimizer = torch.optim.Adam(model.parameters(), lr=10)
    for epoch in range(1000):
        optimizer.zero_grad()
        for x1, x2, pts1, pts2, valid2, cld2 in dataloader:
            loss = model.p2map_loss(x1, x2, pts1, pts2, valid2, cld2)
            loss.backward()
        optimizer.step()
        print(f"epoch {epoch} loss: {loss.item()}")


def test():
    # Note: ensure that H and W is power of 2
    torch.manual_seed(0)
    H, W = 256, 256
    L = 10
    B = 1
    model = FeatMatchNN()
    x1 = torch.rand(B, 3, H, W)
    x2 = torch.rand(B, 3, H, W)
    pts_x = torch.randint(0, H, (B, L, 2))
    pts_y = torch.randint(0, W, (B, L, 2))
    pts1 = torch.stack([pts_x[..., 0], pts_y[..., 0]], dim=-1)
    pts2 = torch.stack([pts_x[..., 1], pts_y[..., 1]], dim=-1)
    valid2 = torch.rand(B, L) > 0.5
    cld2 = torch.rand(B, H, W, 3)
    summary(model, input_data=(x1, x2, pts1))
    # loss = model.p2map_loss(x1, x2, pts1, pts2, valid2, cld2)
    # print(loss)


if __name__ == "__main__":
    main()
    # test()
