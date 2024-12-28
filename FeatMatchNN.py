import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchinfo import summary

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, inset_locator
import os


def viz_frame(frame):
    from DataProcess import viz_frame

    viz_frame(frame)


class FeatMatchDataset(torch.utils.data.Dataset):
    def __init__(self, frames, H=None, W=None, dev=None, augment=True):
        H0, W0 = frames[0][0].shape[-2:]
        self.H = H0 if H is None else H
        self.W = W0 if W is None else W
        x1, x2, pts1, pts2, valid2, cld2 = zip(*frames)
        x1 = torch.stack(x1).to(dev)
        x2 = torch.stack(x2).to(dev)
        pts1 = torch.stack(pts1).to(dev)
        pts2 = torch.stack(pts2).to(dev)
        valid2 = torch.stack(valid2).to(dev)
        cld2 = torch.stack(cld2).to(dev)
        x1, x2, cld2 = v2.functional.resize(
            torch.stack([x1, x2, cld2.permute(0, 3, 1, 2)]), (self.H, self.W)
        )
        cld2 = cld2.permute(0, 2, 3, 1)
        if self.H != H0:
            pts1[..., 0] = pts1[..., 0] * self.H // H0
            pts2[..., 0] = pts2[..., 0] * self.H // H0
        if self.W != W0:
            pts1[..., 1] = pts1[..., 1] * self.W // W0
            pts2[..., 1] = pts2[..., 1] * self.W // W0
        self.x1, self.x2, self.pts1, self.pts2, self.valid2, self.cld2 = (
            x1,
            x2,
            pts1,
            pts2,
            valid2,
            cld2,
        )

        self.transforms = (
            v2.Compose([])
            if not augment
            else v2.Compose(
                [
                    v2.ColorJitter(
                        brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5
                    ),
                    v2.GaussianBlur(3),
                    v2.GaussianNoise(mean=0, sigma=0.03),
                ]
            )
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
        # Note: the same transform for x1 and x2
        x1, x2 = self.transforms(torch.stack([x1, x2]))
        return x1, x2, pts1[:80], pts2[:80], valid2[:80], cld2


# given img1, points1 and img2, return the matched points2 in img2
class FeatMatchNN(nn.Module):
    """Feature Matching Neural Network
    Note: all input and output are torch.Tensor
    """

    def __init__(self):
        super(FeatMatchNN, self).__init__()
        c1, c2, c3, c4, c5 = 3, 4, 8, 16, 16
        # Note whether receptive field is large enough. See https://blog.csdn.net/Rolandxxx/article/details/127270974
        self.conv_enc = nn.ModuleList(
            [
                nn.Conv2d(c1, c2, kernel_size=5, padding=2),
                nn.Conv2d(c2, c3, kernel_size=5, padding=2),
                nn.Conv2d(c3, c4, kernel_size=5, padding=2),
                nn.Conv2d(c4, c5, kernel_size=5, padding=2),
            ]
        )
        self.conv_dec = nn.ModuleList(
            [
                nn.Conv2d(2 * c5 + 1, c4, kernel_size=3, padding=1),
                nn.Conv2d(2 * c4 + 1, c3, kernel_size=3, padding=1),
                nn.Conv2d(2 * c3 + 1, c2, kernel_size=3, padding=1),
                nn.Conv2d(2 * c2 + 1, 1, kernel_size=3, padding=1),
            ]
        )
        self.attns = {}
        self.feats = {}

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
        h1 = x1.view(B, 3, H, W)
        h2 = x2.view(B, 3, H, W)
        pts = pts1.clone().to(torch.int64).view(B, L, 2)
        # encode h1, h2, and calc attn with h1[pts1] and h2 of each layer
        for i, layer in enumerate(self.conv_enc):
            Ci, Hi, Wi = h1.shape[-3:]
            # Conv-ReLU
            h1 = F.leaky_relu(layer(h1))
            h2 = F.leaky_relu(layer(h2))
            # save feats
            self.feats[i] = h2
            # downsample
            h1 = F.max_pool2d(h1, 2)
            h2 = F.max_pool2d(h2, 2)
            # scale indices
            Ci_, Hi_, Wi_ = h1.shape[-3:]
            pts[..., 0] = pts[..., 0] * Hi_ // Hi
            pts[..., 1] = pts[..., 1] * Wi_ // Wi
            # get query that is used to calculate attention, query: (B, Ci_, L)
            query = h1.view(B, Ci_, Hi_ * Wi_).gather(
                2, (pts[:, :, 0] * Wi_ + pts[:, :, 1]).unsqueeze(1).expand(-1, Ci_, -1)
            )

            attn = F.cosine_similarity(
                h2.view(B, Ci_, Hi_ * Wi_, 1).permute(0, 2, 3, 1),
                query.permute(0, 2, 1).unsqueeze(1),
                dim=-1,
            )
            attn = attn.view(B, Hi_, Wi_, L).permute(0, 3, 1, 2)  # (B, L, H, W)

            # attn = torch.einsum("bchw,bcl->blhw", h2, query)

            # self.viz_featmap(attn[0, 0], "attn")
            # self.viz_featmap(h2[0, 3], "h2")
            # self.viz_featmap(h1[0, 3], "h1", pts[0, 0])
            # plt.show()
            self.attns[i] = attn.reshape(B * L, 1, Hi_, Wi_)
        # decode h2 with attns
        h2 = h2.repeat(L, 1, 1, 1)  # (B, Ci, Hi, Wi) -> (B*L, Ci, Hi, Wi)
        for i, layer in enumerate(self.conv_dec):
            idx = len(self.conv_enc) - i - 1
            # concat attn and h2. h2: (B*L, Ci, Hi, Wi), attn: (B*L, 1, Hi, Wi)
            attn = self.attns[idx]
            h2 = torch.concat([h2, attn], dim=1)
            # upsampling
            h2 = F.interpolate(h2, scale_factor=2)
            # concat with self.feats[i]
            feat = self.feats[idx].repeat(L, 1, 1, 1)
            h2 = torch.concat([h2, feat], dim=1)
            # Conv-ReLU
            h2 = F.leaky_relu(layer(h2)) if i < len(self.conv_dec) - 1 else layer(h2)
        # h2: (B*L, 1, H, W) -> (B, L, H, W)
        logits = h2.view(B, L, H, W)
        return logits

    @staticmethod
    def viz_featmap(featmap: torch.Tensor, title="", pts=None):
        # featmap: (H, W)
        fig, ax = plt.subplots()
        cbar = fig.colorbar(
            ax.imshow(featmap.cpu().detach()), ax=ax, orientation="vertical"
        )
        if pts is not None:
            p = pts.cpu().detach()
            ax.scatter(p[..., 1], p[..., 0], c="r", s=4)
        fig.suptitle(title)

    def visualize(self, x1, x2, pts1, pts2, valid2):
        logits = self.forward(x1, x2, pts1)
        B, L, H, W = logits.shape
        img1, img2 = x1[0].permute(1, 2, 0).cpu(), x2[0].permute(1, 2, 0).cpu()
        pt1, pt2, pt2_vis = pts1[0, 0].cpu(), pts2[0, 0].cpu(), int(valid2[0, 0])
        heatmap = (
            F.softmax(logits[0, 0, :, :].view(H * W), dim=-1).view(H, W).cpu().detach()
        )
        fig, axes = plt.subplots(
            1, 2, figsize=(4.5 * 2 * x1.shape[-1] / x1.shape[-2], 4.5)
        )
        axes[0].imshow(img1)
        axes[1].imshow(img2)
        img_heat = axes[1].imshow(heatmap, alpha=0.5, cmap="jet")
        # divider = make_axes_locatable(axes[1])
        # cax = inset_locator.inset_axes(
        #     axes[1],
        #     width="3%",
        #     height="100%",
        #     loc="lower left",
        #     bbox_to_anchor=(1.05, 0.0, 1, 1),
        #     bbox_transform=axes[1].transAxes,
        #     borderpad=0,
        # )
        # cbar = fig.colorbar(img_heat, cax=cax, orientation="vertical")
        axes[0].scatter(pt1[1], pt1[0], c="r", s=4, label="pt1")
        axes[1].scatter(pt2[1], pt2[0], c="r", s=4, label="pt2", alpha=pt2_vis)
        plt.show()

    def loss(self, x1, x2, pts1, pts2, valid2, cld2):
        assert torch.all(valid2), "valid2 should be all True for now"
        logits = self.forward(x1, x2, pts1)
        B, L, H, W = logits.shape
        # get distance
        cld = cld2.view(B, H * W, 3)
        idx = pts2[..., 0] * W + pts2[..., 1]
        pts3d = cld.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))
        dist = torch.norm(cld2.view(B, 1, H*W, 3) - pts3d.view(B, L, 1, 3), dim=-1)
        # get distribution of pts2
        distrib = self.get_distrib(pts2, H, W, sigma=0.3, dist=dist)
        # loss
        loss = F.cross_entropy(logits.view(B * L, H * W), distrib.view(B * L, H * W))
        return loss

    @staticmethod
    def get_distrib(pts, H, W, sigma=0, dist=None):
        """get distribution of points, sigma=0 for hard label (point distrib), >0 for soft label (Gaussian distrib).
        pts: (d1, ..., dN, 2)
        dist: (d1, ..., dN, H, W), distance map from each point to each pixel
        return
            distrib: (d1, ..., dN, H, W)
        """
        batch_dims = pts.shape[:-1]
        pts = pts.view(-1, 2)
        N = pts.shape[0]
        dev = pts.device
        distrib = torch.zeros(N, H, W, device=dev)
        if sigma == 0:
            # point distribution
            distrib[torch.arange(N), pts[:, 0], pts[:, 1]] = 1
        elif sigma > 0:
            if dist is None:
                grid = torch.meshgrid(
                    torch.arange(H, dtype=torch.float, device=dev),
                    torch.arange(W, dtype=torch.float, device=dev),
                    indexing="ij",
                )
                grid = torch.stack(grid, dim=-1)
                dist = torch.norm(grid.view(1, H, W, 2) - pts.view(N, 1, 1, 2), dim=-1)
            # Gaussian distribution
            distrib = torch.exp(-dist.view(N, H, W) / (2 * sigma**2))
            distrib = distrib / torch.sum(distrib, dim=(1, 2), keepdim=True)
        else:
            raise (
                ValueError(
                    "sigma should be non-negative; 0 for hard label, >0 for soft label (Gaussian)"
                )
            )
        pts = pts.view(*batch_dims, 2)
        return distrib.view(*batch_dims, H, W)


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
    try:
        model.load_state_dict(torch.load("fuck.pt", weights_only=True))
    except:
        print("weights not found. fresh train.")
    # train
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1000):
        optimizer.zero_grad()
        for x1, x2, pts1, pts2, valid2, cld2 in dataloader:
            loss = model.loss(x1, x2, pts1, pts2, valid2, cld2)
            loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"\tepoch {epoch} loss: {loss.item():.5f}")
        if (epoch + 1) % 200 == 0:
            model.visualize(x1, x2, pts1, pts2, valid2)
            torch.save(model.state_dict(), "fuck.pt")


def test():
    # Note: ensure that H and W is power of 2
    torch.manual_seed(0)
    H, W = 480, 640
    L = 100
    B = 2
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
