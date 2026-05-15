# InceptionMamba.py
# Windows + PyTorch 2.5.1+cu121 compatible
#
# Optional dependency:
#   pip install mambapy
#
# This file keeps only the 224x224 InceptionMamba-Tiny configuration.

from __future__ import annotations

from typing import List, Tuple, Literal

import torch
import torch.nn as nn


# Optional mambapy backend
try:
    from mambapy.mamba import Mamba, MambaConfig
    HAS_MAMBA_PY = True
except Exception:
    Mamba = None
    MambaConfig = None
    HAS_MAMBA_PY = False


# Build function for your training framework
def build_InceptionMamba_Tiny(num_classes: int, img_channels: int = 3):
    """
    224x224 InceptionMamba-Tiny.

    This is the only exported build function kept for the clean paper-Tiny version.
    """
    return inception_mamba_tiny_224(
        num_classes=num_classes,
        in_chans=img_channels,
    )


# Basic layers
class LayerNorm2d(nn.Module):
    """
    Channel-first LayerNorm for NCHW tensors.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class DropPath(nn.Module):
    """
    Stochastic depth.
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)

        random_tensor = keep_prob + torch.rand(
            shape,
            dtype=x.dtype,
            device=x.device,
        )
        random_tensor.floor_()

        return x.div(keep_prob) * random_tensor


# ConvMixer: 3x3 + 3x11/11x3 + identity
class InceptionBandConvMixer(nn.Module):
    """
    InceptionMamba ConvMixer.

    Channel split:
      square branch: 1/8 C -> depthwise 3x3
      band branch:   1/8 C -> depthwise 3x11 + depthwise 11x3
      identity:      remaining channels
    """
    def __init__(
        self,
        dim: int,
        square_kernel: int = 3,
        band_kernel: int = 11,
        branch_ratio: float = 1.0 / 8.0,
    ):
        super().__init__()

        square_ch = max(1, int(dim * branch_ratio))
        band_ch = max(1, int(dim * branch_ratio))
        identity_ch = dim - square_ch - band_ch

        if identity_ch < 0:
            raise ValueError(f"dim={dim} is too small for branch split.")

        self.square_ch = square_ch
        self.band_ch = band_ch
        self.identity_ch = identity_ch

        self.dw_square = nn.Conv2d(
            square_ch,
            square_ch,
            kernel_size=square_kernel,
            stride=1,
            padding=square_kernel // 2,
            groups=square_ch,
            bias=True,
        )

        self.dw_band_h = nn.Conv2d(
            band_ch,
            band_ch,
            kernel_size=(3, band_kernel),
            stride=1,
            padding=(1, band_kernel // 2),
            groups=band_ch,
            bias=True,
        )

        self.dw_band_v = nn.Conv2d(
            band_ch,
            band_ch,
            kernel_size=(band_kernel, 3),
            stride=1,
            padding=(band_kernel // 2, 1),
            groups=band_ch,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs, xb, xi = torch.split(
            x,
            [self.square_ch, self.band_ch, self.identity_ch],
            dim=1,
        )

        ys = self.dw_square(xs)
        yb = self.dw_band_h(xb) + self.dw_band_v(xb)

        return torch.cat([ys, yb, xi], dim=1)


# Mamba sequence mixer
class FallbackSequenceMixer(nn.Module):
    """
    Fallback only. This is not true Mamba.
    Used only when mambapy is not installed, so the file can still run.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Linear(dim, dim)
        self.mix = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        z = self.norm(x)
        g = torch.sigmoid(self.gate(z))
        y = self.mix(z.transpose(1, 2)).transpose(1, 2)
        return y * g


class MambaSequenceMixer(nn.Module):
    """
    Wrap mambapy Mamba as:
      input:  [B, L, C]
      output: [B, L, C]

    Note:
      mambapy is pure PyTorch and can be memory-heavy.
      The default here uses reduced state settings for Windows/8GB compatibility.
    """
    def __init__(
        self,
        dim: int,
        d_state: int = 8,
        expand_factor: int = 1,
        d_conv: int = 3,
    ):
        super().__init__()

        if HAS_MAMBA_PY:
            try:
                config = MambaConfig(
                    d_model=dim,
                    n_layers=1,
                    d_state=d_state,
                    expand_factor=expand_factor,
                    d_conv=d_conv,
                )
            except TypeError:
                config = MambaConfig(
                    d_model=dim,
                    n_layers=1,
                )

            self.mixer = Mamba(config)
            self.backend = "mambapy"
        else:
            self.mixer = FallbackSequenceMixer(dim)
            self.backend = "fallback"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mixer(x)


# SS2D-like spatial Mamba
class SpatialMamba2D(nn.Module):
    """
    SS2D-like spatial scan using a shared 1D Mamba.

    scan_mode:
      "cross4":
          row-major
          reverse row-major
          column-major
          reverse column-major

      "single":
          row-major only, useful when GPU memory is limited.

    For a clean paper-style Tiny model, "cross4" is the default.
    If you hit CUDA OOM on Windows/mambapy, change the factory to scan_mode="single".
    """
    def __init__(
        self,
        dim: int,
        scan_mode: Literal["single", "cross4"] = "cross4",
        mamba_d_state: int = 8,
        mamba_expand_factor: int = 1,
        mamba_d_conv: int = 3,
    ):
        super().__init__()

        if scan_mode not in ("single", "cross4"):
            raise ValueError("scan_mode must be 'single' or 'cross4'.")

        self.scan_mode = scan_mode

        self.seq_mixer = MambaSequenceMixer(
            dim=dim,
            d_state=mamba_d_state,
            expand_factor=mamba_expand_factor,
            d_conv=mamba_d_conv,
        )

    def _scan_hw(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W] -> [B, H*W, C] -> [B, C, H, W]
        b, c, h, w = x.shape

        seq = x.flatten(2).transpose(1, 2).contiguous()
        out = self.seq_mixer(seq)
        out = out.transpose(1, 2).contiguous().reshape(b, c, h, w)

        return out

    def _scan_wh(self, x: torch.Tensor) -> torch.Tensor:
        # Column-major scan by transposing H/W.
        b, c, h, w = x.shape

        xt = x.transpose(2, 3).contiguous()
        seq = xt.flatten(2).transpose(1, 2).contiguous()

        out = self.seq_mixer(seq)

        out = out.transpose(1, 2).contiguous().reshape(b, c, w, h)
        out = out.transpose(2, 3).contiguous()

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scan_mode == "single":
            return self._scan_hw(x)

        y1 = self._scan_hw(x)

        xr = torch.flip(x, dims=[2, 3])
        y2 = torch.flip(self._scan_hw(xr), dims=[2, 3])

        y3 = self._scan_wh(x)

        xtr = torch.flip(x, dims=[2, 3])
        y4 = torch.flip(self._scan_wh(xtr), dims=[2, 3])

        return (y1 + y2 + y3 + y4) * 0.25


# GlobalMixer with bottleneck Mamba
class BottleneckMambaGlobalMixer(nn.Module):
    """
    GlobalMixer:
      1x1 C -> C/2
      Spatial Mamba
      1x1 C/2 -> C
    """
    def __init__(
        self,
        dim: int,
        bottleneck_ratio: float = 0.5,
        scan_mode: Literal["single", "cross4"] = "cross4",
        mamba_d_state: int = 8,
        mamba_expand_factor: int = 1,
        mamba_d_conv: int = 3,
    ):
        super().__init__()

        hidden_dim = max(1, int(dim * bottleneck_ratio))

        self.reduce = nn.Conv2d(
            dim,
            hidden_dim,
            kernel_size=1,
            bias=True,
        )

        self.ss2d = SpatialMamba2D(
            dim=hidden_dim,
            scan_mode=scan_mode,
            mamba_d_state=mamba_d_state,
            mamba_expand_factor=mamba_expand_factor,
            mamba_d_conv=mamba_d_conv,
        )

        self.expand = nn.Conv2d(
            hidden_dim,
            dim,
            kernel_size=1,
            bias=True,
        )

    @property
    def backend(self) -> str:
        return self.ss2d.seq_mixer.backend

    @property
    def scan_mode(self) -> str:
        return self.ss2d.scan_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reduce(x)
        x = self.ss2d(x)
        x = self.expand(x)
        return x


# InceptionMamba block
class InceptionMambaBlock(nn.Module):
    """
    InceptionMamba block:
      ConvMixer -> GlobalMixer -> Norm -> MLP -> residual
    """
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        conv_branch_ratio: float = 1.0 / 8.0,
        bottleneck_ratio: float = 0.5,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        scan_mode: Literal["single", "cross4"] = "cross4",
        mamba_d_state: int = 8,
        mamba_expand_factor: int = 1,
        mamba_d_conv: int = 3,
    ):
        super().__init__()

        self.conv_mixer = InceptionBandConvMixer(
            dim=dim,
            square_kernel=3,
            band_kernel=11,
            branch_ratio=conv_branch_ratio,
        )

        self.global_mixer = BottleneckMambaGlobalMixer(
            dim=dim,
            bottleneck_ratio=bottleneck_ratio,
            scan_mode=scan_mode,
            mamba_d_state=mamba_d_state,
            mamba_expand_factor=mamba_expand_factor,
            mamba_d_conv=mamba_d_conv,
        )

        self.norm = LayerNorm2d(dim)

        hidden_dim = int(dim * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=True),
        )

        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones(dim),
            requires_grad=True,
        )

        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        x = self.conv_mixer(x)
        x = self.global_mixer(x)
        x = self.norm(x)
        x = self.mlp(x)
        x = x * self.gamma[:, None, None]

        return shortcut + self.drop_path(x)


# Patch embedding / downsampling
class PatchEmbed224(nn.Module):
    """
    224x224 version.

    Produces stage-1 feature map of H/4 x W/4.
    """
    def __init__(self, in_chans: int, embed_dim: int):
        super().__init__()

        mid_dim = embed_dim // 2

        self.proj = nn.Sequential(
            nn.Conv2d(
                in_chans,
                mid_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True,
            ),
            LayerNorm2d(mid_dim),
            nn.GELU(),
            nn.Conv2d(
                mid_dim,
                embed_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True,
            ),
            LayerNorm2d(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Downsample(nn.Module):
    """
    3x3 stride-2 downsampling.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.down = nn.Sequential(
            LayerNorm2d(in_dim),
            nn.Conv2d(
                in_dim,
                out_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


# Backbone
class InceptionMambaBackbone(nn.Module):
    """
    InceptionMamba-Tiny backbone.

    forward_features(x, return_all=True) returns:
      [stage1, stage2, stage3, stage4]

    For 224x224 input:
      stage1: [B, 72, 56, 56]
      stage2: [B, 144, 28, 28]
      stage3: [B, 288, 14, 14]
      stage4: [B, 576, 7, 7]
    """
    def __init__(
        self,
        in_chans: int = 3,
        depths: Tuple[int, int, int, int] = (3, 3, 12, 3),
        dims: Tuple[int, int, int, int] = (72, 144, 288, 576),
        drop_path_rate: float = 0.1,
        mlp_ratio: float = 4.0,
        conv_branch_ratio: float = 1.0 / 8.0,
        bottleneck_ratio: float = 0.5,
        scan_mode: Literal["single", "cross4"] = "cross4",
        mamba_d_state: int = 8,
        mamba_expand_factor: int = 1,
        mamba_d_conv: int = 3,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()

        self.depths = depths
        self.dims = dims
        self.out_channels = dims

        self.patch_embed = PatchEmbed224(in_chans, dims[0])

        total_blocks = sum(depths)
        dp_rates = torch.linspace(0, drop_path_rate, total_blocks).tolist()

        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        cur = 0

        for i in range(4):
            blocks = []

            for _ in range(depths[i]):
                blocks.append(
                    InceptionMambaBlock(
                        dim=dims[i],
                        mlp_ratio=mlp_ratio,
                        conv_branch_ratio=conv_branch_ratio,
                        bottleneck_ratio=bottleneck_ratio,
                        drop_path=dp_rates[cur],
                        layer_scale_init_value=layer_scale_init_value,
                        scan_mode=scan_mode,
                        mamba_d_state=mamba_d_state,
                        mamba_expand_factor=mamba_expand_factor,
                        mamba_d_conv=mamba_d_conv,
                    )
                )
                cur += 1

            self.stages.append(nn.Sequential(*blocks))

            if i < 3:
                self.downsamples.append(
                    Downsample(
                        in_dim=dims[i],
                        out_dim=dims[i + 1],
                    )
                )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @property
    def mamba_backend(self) -> str:
        return self.stages[0][0].global_mixer.backend

    @property
    def scan_mode(self) -> str:
        return self.stages[0][0].global_mixer.scan_mode

    def forward_features(
        self,
        x: torch.Tensor,
        return_all: bool = True,
    ):
        features: List[torch.Tensor] = []

        x = self.patch_embed(x)

        for i in range(4):
            x = self.stages[i](x)
            features.append(x)

            if i < 3:
                x = self.downsamples[i](x)

        if return_all:
            return features

        return x

    def forward(self, x: torch.Tensor):
        return self.forward_features(x, return_all=True)


# Classifier
class InceptionMambaClassifier(nn.Module):
    """
    InceptionMamba-Tiny classifier:
      backbone + global average pooling + LayerNorm + Linear head
    """
    def __init__(
        self,
        num_classes: int = 1000,
        in_chans: int = 3,
        scan_mode: Literal["single", "cross4"] = "cross4",
    ):
        super().__init__()

        self.backbone = InceptionMambaBackbone(
            in_chans=in_chans,
            depths=(3, 3, 12, 3),
            dims=(72, 144, 288, 576),
            drop_path_rate=0.1,
            mlp_ratio=4.0,
            conv_branch_ratio=1.0 / 8.0,
            bottleneck_ratio=0.5,
            scan_mode=scan_mode,
            mamba_d_state=8,
            mamba_expand_factor=1,
            mamba_d_conv=3,
            layer_scale_init_value=1e-6,
        )

        self.norm = nn.LayerNorm(576)
        self.head = nn.Linear(576, num_classes)

    def forward_features(
        self,
        x: torch.Tensor,
        return_all: bool = False,
    ):
        feats = self.backbone.forward_features(x, return_all=True)

        if return_all:
            return feats

        x = feats[-1].mean(dim=(2, 3))
        x = self.norm(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x, return_all=False)
        x = self.head(x)
        return x


# Factory
def inception_mamba_tiny_224(
    num_classes: int = 1000,
    in_chans: int = 3,
    scan_mode: Literal["single", "cross4"] = "cross4",
) -> InceptionMambaClassifier:
    """
    224x224 InceptionMamba-Tiny.

    Args:
        num_classes:
            Number of output classes.
        in_chans:
            Input image channels.
        scan_mode:
            "cross4" is closer to SS2D-style four-direction scan.
            "single" is much more memory-safe on Windows + mambapy.
    """
    return InceptionMambaClassifier(
        num_classes=num_classes,
        in_chans=in_chans,
        scan_mode=scan_mode,
    )


def count_parameters(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


# Random test
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"mambapy available: {HAS_MAMBA_PY}")

    # For maximum paper-style behavior, use scan_mode="cross4".
    # If CUDA OOM occurs on 8GB GPU, change it to scan_mode="single".
    model = inception_mamba_tiny_224(
        num_classes=1000,
        in_chans=3,
        scan_mode="cross4",
    ).to(device)

    model.eval()

    x = torch.randn(1, 3, 224, 224, device=device)

    with torch.no_grad():
        y = model(x)
        feats = model.forward_features(x, return_all=True)

    print("\n[InceptionMamba-Tiny 224x224]")
    print(f"Mamba backend: {model.backbone.mamba_backend}")
    print(f"Scan mode: {model.backbone.scan_mode}")
    print(f"Params: {count_parameters(model):.2f} M")
    print(f"Input:  {tuple(x.shape)}")
    print(f"Output: {tuple(y.shape)}")

    for i, f in enumerate(feats, 1):
        print(f"Stage {i}: {tuple(f.shape)}")

    assert y.shape == (1, 1000)

    print("\nRandom test passed.")