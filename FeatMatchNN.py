import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchinfo import summary

import os

import matplotlib.pyplot as plt
import matplotlib.patches
from mpl_toolkits.axes_grid1 import make_axes_locatable, inset_locator

import cv2
import numpy as np


def viz_frame(frame):
    from DataProcess import viz_frame

    viz_frame(frame)


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


def Img2Tensor(img):
    # img: (B, H, W, 3), np.uint8
    return torch.tensor(img).permute(0, 3, 1, 2) / 255


def Tensor2Img(x):
    # x: (B, 3, H, W), torch.float32, 0~1
    return (x * 255).permute(0, 2, 3, 1).cpu().detach().numpy().astype(np.uint8)


class MatchPointFigure:
    def __init__(self, img1, img2):
        self.H, self.W = img1.shape[:2]
        self.fig, self.axes = plt.subplots(
            1, 2, figsize=(4.5 * 2 * self.W / self.H, 4.5)
        )
        self.axes[0].imshow(img1)
        self.axes[1].imshow(img2)
        self.p1 = plt.Circle([0, 0], 1, color="r")
        self.p2 = plt.Circle([0, 0], 1, color="r")
        self.conn = matplotlib.patches.ConnectionPatch(
            xyA=(0, 0),
            xyB=(0, 0),
            coordsA=self.axes[0].transData,
            coordsB=self.axes[1].transData,
            color="r",
        )
        self.heatmap = None
        # add artists to figure
        self.axes[0].add_artist(self.p1)
        self.axes[1].add_artist(self.p2)
        self.fig.add_artist(self.conn)
        self.p1.set_visible(False)
        self.p2.set_visible(False)
        self.conn.set_visible(False)
        return

    def refresh(self, m_uv1, m_uv2, valid, heatmap=None, suptitle=None):
        self.p1.set_visible(True)
        self.p2.set_visible(valid)
        self.conn.set_visible(valid)
        self.p1.center = m_uv1
        if valid:
            self.p2.center = m_uv2
            self.conn.xy1 = m_uv1
            self.conn.xy2 = m_uv2
        if heatmap is not None:
            if self.heatmap is None:
                self.heatmap = self.axes[1].imshow(heatmap, alpha=0.5, cmap="jet")
            else:
                self.heatmap.set_array(heatmap)
        if suptitle is not None:
            self.fig.suptitle(suptitle)
        self.fig.canvas.draw()


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
        x1 = v2.functional.gaussian_noise(x1, mean=0, sigma=0.01)
        x2 = v2.functional.gaussian_noise(x2, mean=0, sigma=0.01)
        return x1, x2, pts1, pts2, valid2, cld2


# given img1, points1 and img2, return the matched points2 in img2
class FeatMatchNN(nn.Module):
    """Feature Matching Neural Network
    Note: all input and output are torch.Tensor
    """

    def __init__(self, device):
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
        # Kq
        self.Kq = nn.ModuleList(
            [
                nn.Linear(c2, 2 * c2),
                nn.Linear(c3, 2 * c3),
                nn.Linear(c4, 2 * c4),
                nn.Linear(c5, 2 * c5),
            ]
        )
        # Kv
        self.Kv = nn.ModuleList(
            [
                nn.Linear(c2, 2 * c2),
                nn.Linear(c3, 2 * c3),
                nn.Linear(c4, 2 * c4),
                nn.Linear(c5, 2 * c5),
            ]
        )

        self.conv_dec = nn.ModuleList(
            [
                nn.Conv2d(2 * c5 + 1, c4, kernel_size=3, padding=1),
                nn.Conv2d(2 * c4 + 1, c3, kernel_size=3, padding=1),
                nn.Conv2d(2 * c3 + 1, c2, kernel_size=3, padding=1),
                nn.Conv2d(2 * c2 + 1, c1, kernel_size=3, padding=1),
            ]
        )
        self.conv_output = nn.Conv2d(c1, 1, kernel_size=3, padding=1)
        self.attns = {}
        self.feats = {}
        self.thresh_valid = 10
        self.device = device
        self.to(device)

    def forward(self, x1, x2, pts1):
        """
        x1, x2: (B, 3, H, W), float, 0~1
        pts1: (B, L, 2), int

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
            attn = torch.einsum(
                "blc,bhwc->blhw",
                self.Kq[i](query.permute(0, 2, 1)),
                self.Kv[i](h2.permute(0, 2, 3, 1)),
            )
            # attn = F.softmax(attn.view(B*L, -1), dim=-1)

            # viz_featmap(attn[0, 90], "attn")
            # viz_featmap(h2[0, 1], "h2")
            # viz_featmap(h1[0, 1], "h1", pts[0, 90])
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
            h2 = F.leaky_relu(layer(h2))
        # output layer
        logits = self.conv_output(h2).view(B, L, H, W)
        return logits

    def loss(self, x1, x2, pts1, pts2, valid2, cld2, sigma=0.1):
        """loss function for feature matching, including precision and misdetection
        ---
        x1, x2: (B, 3, H, W), float, 0~1
        pts1, pts2: (B, L, 2), int
        valid2: (B, L), boolean
        cld2: (B, H, W, 3)
        sigma: float, sigma=0 for hard label (point distrib), >0 for soft label (3D Gaussian distrib), recommended schedule when training from scratch: 0.5-0.3-0.1-0.05
        return
            loss: scalar
        """
        # assert torch.all(valid2), "valid2 should be all True for now"
        logits = self.forward(x1, x2, pts1)
        dev = x1.device
        B, L, H, W = logits.shape
        logits_m = logits[valid2]  # (B, L, H, W) -> (N1, H, W)
        logits_um = logits[~valid2]  # (B, L, H, W) -> (N2, H, W)

        """ Part 1: loss for regression precision """
        # get distance
        cld = cld2.view(B, H * W, 3)
        idx = pts2[..., 0] * W + pts2[..., 1]
        pts3d = cld.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))
        dist = torch.norm(cld2.view(B, 1, H * W, 3) - pts3d.view(B, L, 1, 3), dim=-1)
        # select valid points
        pts = pts2.view(B * L, 2)[valid2.view(-1)]  # (B, L, 2) -> (N1, 2)
        dist = dist.view(B * L, H, W)[valid2.view(-1)]  # (B, L, H, W) -> (N1, H, W)
        # get distribution of pts2
        distrib = self.get_distrib(pts, H, W, sigma=0.05, dist=dist)
        loss_prec = F.cross_entropy(logits_m.view(-1, H * W), distrib.view(-1, H * W))

        """ Part 2: loss for misdetection """
        # get invalid points (unmatched case)
        max_um = torch.max(logits_um.flatten(-2), dim=-1)[0]
        max_m = torch.max(logits_m.flatten(-2), dim=-1)[0]
        loss_misdet = (
            F.relu(max_um - 0).sum() + F.relu(self.thresh_valid - max_m).sum()
        ) / (B * L)

        # valid_pred = torch.max(logits.flatten(-2), dim=-1)[0] > 0
        # print(
        #     f"incorrect validity prediction: {torch.bitwise_xor(valid_pred, valid2).sum()}"
        # )

        return loss_prec + loss_misdet

    def viz_pred(self, x1, x2, pts1, pts2, valid2):
        """
        x1, x2: (3, H, W), float, 0~1
        pts1, pts2: (L, 2), int
        valid2: (L, ), bool
        """
        assert x1.dim() == 3 and x2.dim() == 3 and pts1.dim() == 2 and pts2.dim() == 2
        H, W = x1.shape[-2:]
        L = pts1.shape[-2]
        logits = self.forward(x1.unsqueeze(0), x2.unsqueeze(0), pts1.unsqueeze(0))
        heatmaps = F.softmax(logits.view(L, H * W), dim=-1).view(L, H, W).cpu().detach()
        img1 = x1.permute(1, 2, 0).cpu().detach()
        img2 = x2.permute(1, 2, 0).cpu().detach()
        uv1 = pts1.cpu().detach().flip([-1])
        uv2 = pts2.cpu().detach().flip([-1])
        valid = valid2.cpu().detach()

        mpfig = MatchPointFigure(img1, img2)
        idx = 0
        mpfig.refresh(
            uv1[idx],
            uv2[idx],
            valid[idx],
            heatmaps[idx],
            f"Feature Matching Prediction {idx} / {L}",
        )

        def on_key(event):
            nonlocal mpfig, idx
            if event.key == "q":
                plt.close()
                return
            if event.key == "n":
                idx += 1 if idx < L - 1 else 0
            elif event.key == "m":
                idx -= 1 if idx > 0 else 0
            mpfig.refresh(
                uv1[idx],
                uv2[idx],
                valid[idx],
                heatmaps[idx],
                f"Feature Matching Prediction {idx} / {L}",
            )

        mpfig.fig.canvas.mpl_connect("key_press_event", on_key)
        plt.show()

    def match(self, img1, img2):
        """match points in img1 to img2
        img1, img2: (H, W, 3), np.uint8 or (3, H, W), torch.float32, 0~1
        """
        # if img1 is np array, convert to torch.Tensor
        if isinstance(img1, np.ndarray):
            H, W = img1.shape[:2]
            x1 = Img2Tensor(img1).unsqueeze(0).to(self.device)
            x2 = Img2Tensor(img2).unsqueeze(0).to(self.device)
            mpfig = MatchPointFigure(img1, img2)
        else:
            H, W = img1.shape[-2:]
            x1 = img1.unsqueeze(0).to(self.device)
            x2 = img2.unsqueeze(0).to(self.device)
            mpfig = MatchPointFigure(
                img1.permute(1, 2, 0).cpu().detach(),
                img2.permute(1, 2, 0).cpu().detach(),
            )

        def on_click(event):
            nonlocal mpfig
            if event.inaxes is None:
                return
            u1, v1 = int(event.xdata), int(event.ydata)
            print(f"clicked: {u1}, {v1}", end="\t")
            pts1 = torch.tensor([v1, u1]).view(1, 1, 2).to(self.device)
            logits = self.forward(x1, x2, pts1).view(H * W).cpu().detach()
            heatmap = F.softmax(logits, dim=-1).view(H, W)
            idx = torch.argmax(logits)
            u2, v2 = idx % W, idx // W
            valid = logits.max() > self.thresh_valid / 2
            mpfig.refresh(
                (u1, v1), (u2, v2), valid, heatmap, "Feature Matching Prediction"
            )
            print(f"valid: {valid}")

        mpfig.fig.canvas.mpl_connect("button_press_event", on_click)
        plt.show()

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
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    # create model
    model = FeatMatchNN(dev)
    try:
        model.load_state_dict(torch.load("fuck.pt", weights_only=True))
    except:
        print("weights not found. fresh train.")
    # train
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    schedule = [[1000, 0.5], [2000, 0.3], [3000, 0.1], [5000, 0.05]]
    for epoch in range(3000):
        optimizer.zero_grad()
        # get scheduled sigma
        for e, s in schedule:
            if epoch < e:
                break
        # Batch Gradient Descent
        for x1, x2, pts1, pts2, valid2, cld2 in dataloader:
            loss = model.loss(x1, x2, pts1, pts2, valid2, cld2, s)
            loss.backward()
            break
        # model.viz_pred(x1[0], x2[0], pts1[0], pts2[0], valid2[0])
        model.match(x1[0], x2[0])
        optimizer.step()
        if epoch % 20 == 0:
            print(f"\tepoch {epoch} loss: {loss.item():.5f}")
        if (epoch + 1) % 200 == 0:
            torch.save(model.state_dict(), "fuck.pt")


def test():
    # Note: ensure that H and W is power of 2
    torch.manual_seed(0)
    H, W = 128, 128
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
    # print(loss)


if __name__ == "__main__":
    main()
    # test()

    # img1 = cv2.imread("3.jpg")
    # img2 = cv2.imread("4.jpg")
    # img1 = cv2.resize(img1, (128, 128))
    # img2 = cv2.resize(img2, (128, 128))

    # model = FeatMatchNN("cpu")
    # model.match(img1, img2)
