# inception_mamba_small.py
# Windows / PyTorch 2.5.1+cu121 compatible
# pip install mambapy einops

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from mambapy.mamba import Mamba, MambaConfig
    HAS_MAMBAPY = True
except Exception:
    HAS_MAMBAPY = False


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device
        )
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class LayerNorm2d(nn.Module):
    """Channel-first LayerNorm for NCHW."""
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        # NCHW -> NHWC -> NCHW
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        act: bool = True,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.GELU() if act else nn.Identity(),
        )

    def forward(self, x):
        return self.net(x)


class LargeBandDWConv(nn.Module):
    """
    InceptionMamba-style local mixer.
    Similar to the large band depthwise convolution in the paper:
    - square depthwise conv
    - horizontal band depthwise conv
    - vertical band depthwise conv
    - identity branch
    Input/Output: N, C, H, W
    """

    def __init__(
        self,
        dim: int,
        square_kernel: int = 3,
        band_kernel: int = 11,
        branch_ratio: float = 0.25,
    ):
        super().__init__()
        branch_ch = max(1, int(dim * branch_ratio))

        # Some channels are reserved for identity processing, while the remaining channels are distributed across three DW conv branches.
        conv_ch = branch_ch * 3
        if conv_ch > dim:
            raise ValueError("The branch_ratio is too large, causing the conv branch channel to exceed dim.")

        self.id_ch = dim - conv_ch
        self.branch_ch = branch_ch

        self.dw_square = nn.Conv2d(
            branch_ch,
            branch_ch,
            kernel_size=square_kernel,
            padding=square_kernel // 2,
            groups=branch_ch,
            bias=True,
        )

        self.dw_h = nn.Conv2d(
            branch_ch,
            branch_ch,
            kernel_size=(1, band_kernel),
            padding=(0, band_kernel // 2),
            groups=branch_ch,
            bias=True,
        )

        self.dw_v = nn.Conv2d(
            branch_ch,
            branch_ch,
            kernel_size=(band_kernel, 1),
            padding=(band_kernel // 2, 0),
            groups=branch_ch,
            bias=True,
        )

        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

    def forward(self, x):
        x_id, x_sq, x_h, x_v = torch.split(
            x,
            [self.id_ch, self.branch_ch, self.branch_ch, self.branch_ch],
            dim=1,
        )

        out = torch.cat(
            [
                x_id,
                self.dw_square(x_sq),
                self.dw_h(x_h),
                self.dw_v(x_v),
            ],
            dim=1,
        )

        return self.proj(out)


class FallbackGlobalMixer(nn.Module):
    """
    The fallback without mambapy.
    This isn't a proper Mamba implementation; it's just to ensure the program can run the shape test on Windows.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x):
        return self.net(x)


class MambaBottleneck2d(nn.Module):
    """
    2D feature map -> sequence -> Mamba -> 2D feature map
    Input: N, C, H, W
    Convert to: N, H*W, hidden_dim
    Output: N, C, H, W
    """

    def __init__(
        self,
        dim: int,
        bottleneck_ratio: float = 0.25,
        use_mamba: bool = True,
    ):
        super().__init__()
        hidden_dim = max(16, int(dim * bottleneck_ratio))

        self.in_proj = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

        self.using_real_mamba = bool(use_mamba and HAS_MAMBAPY)

        if self.using_real_mamba:
            config = MambaConfig(
                d_model=hidden_dim,
                n_layers=1,
            )
            self.mixer = Mamba(config)
        else:
            self.mixer = FallbackGlobalMixer(hidden_dim)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.in_proj(x)                       # B, D, H, W
        x = x.flatten(2).transpose(1, 2)          # B, L, D
        x = self.norm(x)
        x = self.mixer(x)                         # B, L, D
        x = x.transpose(1, 2).reshape(b, -1, h, w)
        x = self.out_proj(x)

        return x


class InceptionMambaBlock(nn.Module):
    """
    InceptionMamba-style block:
    local large-band DW conv mixer + Mamba bottleneck global mixer + MLP
    """

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        band_kernel: int = 11,
        bottleneck_ratio: float = 0.25,
        use_mamba: bool = True,
    ):
        super().__init__()

        self.norm1 = LayerNorm2d(dim)
        self.local_mixer = LargeBandDWConv(
            dim=dim,
            square_kernel=3,
            band_kernel=band_kernel,
            branch_ratio=0.25,
        )

        self.norm2 = LayerNorm2d(dim)
        self.global_mixer = MambaBottleneck2d(
            dim=dim,
            bottleneck_ratio=bottleneck_ratio,
            use_mamba=use_mamba,
        )

        self.norm3 = LayerNorm2d(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, dim, kernel_size=1),
        )

        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path(self.local_mixer(self.norm1(x)))
        x = x + self.drop_path(self.global_mixer(self.norm2(x)))
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x


class DownsampleLayer(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, first: bool = False):
        super().__init__()

        if first:
            # 224 -> 56
            self.net = nn.Sequential(
                ConvBNAct(in_ch, out_ch // 2, kernel_size=3, stride=2),
                ConvBNAct(out_ch // 2, out_ch, kernel_size=3, stride=2),
            )
        else:
            # H, W -> H/2, W/2
            self.net = nn.Sequential(
                LayerNorm2d(in_ch),
                nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2),
            )

    def forward(self, x):
        return self.net(x)


class InceptionMambaBackbone(nn.Module):
    """
    Feature layer.
    With a default input of 224x224, the output stages are:
    - stage 1: B, 32, 56, 56
    - stage 2: B, 64, 28, 28
    - stage 3: B, 128, 14, 14
    - stage 4: B, 256, 7, 7
    This multi-scale output is convenient for connecting to object detection, semantic segmentation, or image caption encoders.
    """

    def __init__(
        self,
        in_chans: int = 3,
        dims=(32, 64, 128, 256),
        depths=(2, 2, 4, 2),
        drop_path_rate: float = 0.05,
        use_mamba: bool = True,
    ):
        super().__init__()

        assert len(dims) == 4
        assert len(depths) == 4

        self.dims = dims
        self.depths = depths

        self.downsamples = nn.ModuleList()
        self.downsamples.append(
            DownsampleLayer(in_chans, dims[0], first=True)
        )

        for i in range(3):
            self.downsamples.append(
                DownsampleLayer(dims[i], dims[i + 1], first=False)
            )

        total_blocks = sum(depths)
        dp_rates = torch.linspace(0, drop_path_rate, total_blocks).tolist()

        self.stages = nn.ModuleList()
        cur = 0
        for stage_idx in range(4):
            blocks = []
            for _ in range(depths[stage_idx]):
                blocks.append(
                    InceptionMambaBlock(
                        dim=dims[stage_idx],
                        mlp_ratio=4.0,
                        drop_path=dp_rates[cur],
                        band_kernel=11,
                        bottleneck_ratio=0.25,
                        use_mamba=use_mamba,
                    )
                )
                cur += 1

            self.stages.append(nn.Sequential(*blocks))

        self.out_channels = list(dims)

    def forward(self, x, return_stages: bool = True):
        features = []

        for down, stage in zip(self.downsamples, self.stages):
            x = down(x)
            x = stage(x)
            features.append(x)

        if return_stages:
            return features

        return features[-1]


class InceptionMambaClassifier(nn.Module):
    """
    Classification model = backbone + classification head
    """

    def __init__(
        self,
        num_classes: int = 1000,
        in_chans: int = 3,
        dims=(32, 64, 128, 256),
        depths=(2, 2, 4, 2),
        drop_path_rate: float = 0.05,
        use_mamba: bool = True,
    ):
        super().__init__()

        self.backbone = InceptionMambaBackbone(
            in_chans=in_chans,
            dims=dims,
            depths=depths,
            drop_path_rate=drop_path_rate,
            use_mamba=use_mamba,
        )

        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)

    def forward_features(self, x, return_stages: bool = False):
        feats = self.backbone(x, return_stages=True)

        if return_stages:
            return feats

        x = feats[-1]
        x = x.mean(dim=(2, 3))       # global average pooling
        x = self.norm(x)
        return x

    def forward_head(self, x):
        return self.head(x)

    def forward(self, x):
        x = self.forward_features(x, return_stages=False)
        x = self.forward_head(x)
        return x


def count_params(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {device}")
    print(f"mambapy available: {HAS_MAMBAPY}")

    model = InceptionMambaClassifier(
        num_classes=10,
        dims=(32, 64, 128, 256),
        depths=(1, 1, 2, 1),   # Small beta version; the official version can be changed to (2, 2, 4, 2)
        drop_path_rate=0.05,
        use_mamba=True,
    ).to(device)

    model.eval()

    x = torch.randn(2, 3, 224, 224, device=device)

    with torch.no_grad():
        logits = model(x)
        feats = model.forward_features(x, return_stages=True)

    print(f"Params: {count_params(model) / 1e6:.3f} M")
    print(f"Input:  {tuple(x.shape)}")
    print(f"Logits: {tuple(logits.shape)}")

    for i, f in enumerate(feats, start=1):
        print(f"Stage {i}: {tuple(f.shape)}")

    assert logits.shape == (2, 10)
    assert feats[0].shape[2:] == (56, 56)
    assert feats[1].shape[2:] == (28, 28)
    assert feats[2].shape[2:] == (14, 14)
    assert feats[3].shape[2:] == (7, 7)

    print("Random forward test passed.")