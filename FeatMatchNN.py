import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchinfo import summary

import matplotlib.pyplot as plt
import os


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
        x1, x2, pts1, pts2, valid2, cld2 = (
            self.x1[idx],
            self.x2[idx],
            self.pts1[idx],
            self.pts2[idx],
            self.valid2[idx],
            self.cld2[idx],
        )
        H0, W0 = x1.shape[-2:]
        x1, x2, cld2 = v2.functional.resize(
            torch.stack([x1, x2, cld2.permute(2, 0, 1)]), (self.H, self.W)
        )
        cld2 = cld2.permute(1, 2, 0)
        # # Note: the same transform for x1 and x2
        x1, x2 = self.transforms(torch.stack([x1, x2]))
        if self.H != H0:
            pts1[..., 0] = pts1[..., 0] * self.H // H0
            pts2[..., 0] = pts2[..., 0] * self.H // H0
        if self.W != W0:
            pts1[..., 1] = pts1[..., 1] * self.W // W0
            pts2[..., 1] = pts2[..., 1] * self.W // W0
        return x1, x2, pts1[:80], pts2[:80], valid2[:80], cld2


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
        self.attns = {}

    def forward(self, x1, x2, pts1):
        """
        x1, x2: (B, 3, H, W), elements in [0, 1] instead of [0, 255]
        pts1: (B, L, 2)

        return
            logits: (B, L, H, W), unscaled confidence map of x2, elements in (-inf, inf)
        """
        B, L = pts1.shape[:2]
        H, W = x1.shape[-2:]
        # copy and reshape
        h1 = x1.clone().detach().view(B, 3, H, W)
        h2 = x2.clone().detach().view(B, 3, H, W)
        pts = pts1.clone().detach().to(torch.int64).view(B, L, 2)
        # encode h1, h2, and calc attn with h1[pts1] and h2 of each layer
        for i, layer in enumerate(self.conv_enc):
            Ci, Hi, Wi = h1.shape[-3:]
            # Conv-ReLU-Pool once
            h1 = F.relu(layer(h1))
            h2 = F.relu(layer(h2))
            h1 = F.max_pool2d(h1, 2)
            h2 = F.max_pool2d(h2, 2)
            # scale indices
            Ci_, Hi_, Wi_ = h1.shape[-3:]
            pts[..., 0] = pts[..., 0] * Hi_ // Hi
            pts[..., 1] = pts[..., 1] * Wi_ // Wi
            # get Q that is used to calculate attention, Q: (B, Ci_, L)
            # Q[i, j, k] = h1[i, j, pts[i, k, 0], pts[i, k, 1]]
            Q = h1.view(B, Ci_, Hi_ * Wi_).gather(
                2, (pts[:, :, 0] * Wi_ + pts[:, :, 1]).view(B, 1, L)
            )
            # attn = F.sigmoid(torch.einsum("bchw,bcl->blhw", h2, Q))
            # attns.append(attn.reshape(-1, 1, Hi_, Wi_))
            attn = F.softmax(
                torch.einsum("bchw,bcl->blhw", h2, Q).view(B, L, -1), dim=-1
            )
            self.attns[i] = attn.view(B * L, 1, Hi_, Wi_)
        # decode h2 with attns
        h2 = h2.repeat(L, 1, 1, 1)  # (B, Ci, Hi, Wi) -> (B*L, Ci, Hi, Wi)
        for i, layer in enumerate(self.conv_dec):
            attn = self.attns[len(self.conv_enc) - i - 1]
            # h2: (B*L, Ci, Hi, Wi), attn: (B*L, 1, Hi, Wi)
            h2 = layer(h2 * attn)
            if i < len(self.conv_dec) - 1:
                h2 = F.relu(h2)
        # h2: (B*L, 1, H, W), logits: (B, L, H, W)
        logits = h2.view(B, L, H, W)
        return logits

    def visualize(self, x1, x2, pts1, pts2, valid2):
        logits = self.forward(x1, x2, pts1)
        img1, img2 = x1[0].permute(1, 2, 0).cpu(), x2[0].permute(1, 2, 0).cpu()
        pt1, pt2, pt2_vis = pts1[0, 0].cpu(), pts2[0, 0].cpu(), int(valid2[0, 0])
        heatmap = F.sigmoid(logits[0, 0, :, :]).cpu().detach()
        fig, axes = plt.subplots(
            1, 2, figsize=(4.5 * 2 * x1.shape[-1] / x1.shape[-2], 4.5)
        )
        axes[0].imshow(img1)
        axes[1].imshow(img2)
        img_heat = axes[1].imshow(heatmap, alpha=0.5, cmap="jet")
        # img_heat.set_clim(0, 1)
        cbar = fig.colorbar(img_heat, ax=axes[1], orientation="vertical")
        axes[0].scatter(pt1[1], pt1[0], c="r", s=4, label="pt1")
        axes[1].scatter(pt2[1], pt2[0], c="r", s=4, label="pt2", alpha=pt2_vis)
        plt.show()

    def fuck_loss(self, x1, x2, pts1, pts2, valid2):
        logits = self.forward(x1, x2, pts1)
        B, L, H, W = logits.shape
        prob_map = F.sigmoid(logits)  # (B, L, H, W)
        probs = (
            prob_map.view(B, L, -1)
            .gather(-1, (pts2[..., 0] * W + pts2[..., 1]).view(B, L, 1))
            .view(B, L)
        )  # (B, L)
        l1 = (~valid2) * torch.sum(prob_map.view(B, L, -1), -1)
        l2 = valid2 * (1 - probs)
        return torch.mean(l1 + l2)


def main():
    dev = torch.device("cuda")
    # load data
    dataset_dir = os.path.join(os.path.dirname(__file__), "dataset")
    frames = []
    for f in os.listdir(dataset_dir):
        frames += torch.load(f"{dataset_dir}/{f}", weights_only=True)
        print(f"{f} loaded.")
    # create dataset
    dataset = FeatMatchDataset(frames, H=480, W=640, dev=dev)
    # create dataloader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    # create model
    model = FeatMatchNN().to(dev)
    try:
        model.load_state_dict(torch.load("fuck_featmatch.pt", weights_only=True))
    except:
        print("weights not found. fresh train.")
    # train
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
    for epoch in range(1000):
        optimizer.zero_grad()
        for x1, x2, pts1, pts2, valid2, cld2 in dataloader:
            loss = model.fuck_loss(x1, x2, pts1, pts2, valid2)
            loss.backward()
        optimizer.step()
        print(f"epoch {epoch} loss: {loss.item():.3f}")
        if epoch % 100 == 0:
            model.visualize(x1, x2, pts1, pts2, valid2)
            torch.save(model.state_dict(), "fuck_featmatch.pt")


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
