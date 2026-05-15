# InceptionMamba.py
# Windows + PyTorch 2.5.1+cu121 compatible
# Optional:
#   pip install mambapy

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


# Build functions for your training framework
def build_InceptionMamba_Tiny(num_classes, img_channels=3):
    return inception_mamba_tiny_224_paper(
        num_classes=num_classes,
        in_chans=img_channels,
        scan_mode="single",
    )


def build_InceptionMamba_Tiny32(num_classes, img_channels=3):
    """32x32 CIFAR-style reduced version."""
    return inception_mamba_cifar_32(
        num_classes=num_classes,
        in_chans=img_channels,
    )


# Basic layers
class LayerNorm2d(nn.Module):
    """Channel-first LayerNorm for NCHW tensors."""
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
    """Stochastic depth."""
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
      square branch: 1/8 C -> DWConv 3x3
      band branch:   1/8 C -> DWConv 3x11 + DWConv 11x3
      identity:      6/8 C -> identity
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


# Fallback sequence mixer
class FallbackSequenceMixer(nn.Module):
    """
    Fallback only. This is not true Mamba.
    Used when mambapy is not installed.
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


# Mamba sequence mixer
class MambaSequenceMixer(nn.Module):
    """
    Wrap mambapy Mamba as:
      input:  [B, L, C]
      output: [B, L, C]

    mambapy 是 pure PyTorch，selective_scan 會建立較大的中間張量。
    8GB GPU 建議：
      d_state=8
      expand_factor=1
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
                # For older mambapy versions.
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


# Spatial Mamba 2D
class SpatialMamba2D(nn.Module):
    """
    Spatial Mamba for feature maps.

    scan_mode:
      "single":
          Row-major single scan.
          Recommended for Windows + mambapy + 8GB GPU.

      "cross4":
          Four-direction scan approximation:
            1. row-major
            2. reverse row-major
            3. column-major
            4. reverse column-major

          This is closer to SS2D cross-scan concept, but it is much heavier.
    """
    def __init__(
        self,
        dim: int,
        scan_mode: Literal["single", "cross4"] = "single",
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
        b, c, h, w = x.shape

        seq = x.flatten(2).transpose(1, 2).contiguous()
        # [B, H*W, C]

        out = self.seq_mixer(seq)

        out = out.transpose(1, 2).contiguous().reshape(b, c, h, w)
        return out

    def _scan_wh(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        xt = x.transpose(2, 3).contiguous()
        seq = xt.flatten(2).transpose(1, 2).contiguous()
        # [B, W*H, C]

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
      1x1 C -> C * bottleneck_ratio
      Spatial Mamba
      1x1 hidden -> C
    """
    def __init__(
        self,
        dim: int,
        bottleneck_ratio: float = 0.5,
        scan_mode: Literal["single", "cross4"] = "single",
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


# InceptionMamba Block
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
        scan_mode: Literal["single", "cross4"] = "single",
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
    Produces H/4 x W/4 stage-1 feature map.
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


class PatchEmbed32(nn.Module):
    """
    32x32 version.
    No early downsampling.
    """
    def __init__(self, in_chans: int, embed_dim: int):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(
                in_chans,
                embed_dim,
                kernel_size=3,
                stride=1,
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
    Feature extractor only.

    forward_features(x, return_all=True) returns:
      [stage1, stage2, stage3, stage4]
    """
    def __init__(
        self,
        in_chans: int = 3,
        img_size_mode: Literal["224", "32"] = "224",
        depths: Tuple[int, int, int, int] = (3, 3, 12, 3),
        dims: Tuple[int, int, int, int] = (72, 144, 288, 576),
        drop_path_rate: float = 0.1,
        mlp_ratio: float = 4.0,
        conv_branch_ratio: float = 1.0 / 8.0,
        bottleneck_ratio: float = 0.5,
        scan_mode: Literal["single", "cross4"] = "single",
        mamba_d_state: int = 8,
        mamba_expand_factor: int = 1,
        mamba_d_conv: int = 3,
    ):
        super().__init__()

        self.depths = depths
        self.dims = dims
        self.out_channels = dims
        self.img_size_mode = img_size_mode

        if img_size_mode == "224":
            self.patch_embed = PatchEmbed224(in_chans, dims[0])
        elif img_size_mode == "32":
            self.patch_embed = PatchEmbed32(in_chans, dims[0])
        else:
            raise ValueError("img_size_mode must be '224' or '32'.")

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
        block = self.stages[0][0]
        return block.global_mixer.backend

    @property
    def scan_mode(self) -> str:
        block = self.stages[0][0]
        return block.global_mixer.scan_mode

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
    Classification model:
      backbone + global average pooling + LayerNorm + Linear head
    """
    def __init__(
        self,
        num_classes: int = 1000,
        in_chans: int = 3,
        img_size_mode: Literal["224", "32"] = "224",
        depths: Tuple[int, int, int, int] = (3, 3, 12, 3),
        dims: Tuple[int, int, int, int] = (72, 144, 288, 576),
        drop_path_rate: float = 0.1,
        mlp_ratio: float = 4.0,
        conv_branch_ratio: float = 1.0 / 8.0,
        bottleneck_ratio: float = 0.5,
        scan_mode: Literal["single", "cross4"] = "single",
        mamba_d_state: int = 8,
        mamba_expand_factor: int = 1,
        mamba_d_conv: int = 3,
    ):
        super().__init__()

        self.backbone = InceptionMambaBackbone(
            in_chans=in_chans,
            img_size_mode=img_size_mode,
            depths=depths,
            dims=dims,
            drop_path_rate=drop_path_rate,
            mlp_ratio=mlp_ratio,
            conv_branch_ratio=conv_branch_ratio,
            bottleneck_ratio=bottleneck_ratio,
            scan_mode=scan_mode,
            mamba_d_state=mamba_d_state,
            mamba_expand_factor=mamba_expand_factor,
            mamba_d_conv=mamba_d_conv,
        )

        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)

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


# Factory: 224x224 paper Tiny
def inception_mamba_tiny_224_paper(
    num_classes: int = 1000,
    in_chans: int = 3,
    scan_mode: Literal["single", "cross4"] = "single",
) -> InceptionMambaClassifier:
    """
    224x224 InceptionMamba-Tiny paper configuration.

    Paper Tiny:
      depths = (3, 3, 12, 3)
      dims   = (72, 144, 288, 576)

    scan_mode:
      "single" recommended on 8GB GPU.
      "cross4" closer to SS2D cross-scan but heavier.
    """
    return InceptionMambaClassifier(
        num_classes=num_classes,
        in_chans=in_chans,
        img_size_mode="224",
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
    )


# Factory: 32x32 CIFAR reduced model
def inception_mamba_cifar_32(
    num_classes: int = 10,
    in_chans: int = 3,
) -> InceptionMambaClassifier:
    """
    32x32 CIFAR-style reduced InceptionMamba.

    This is not the paper Tiny model.
    It is a small-image friendly design.

    Reduced config:
      depths = (1, 1, 3, 1)
      dims   = (32, 64, 128, 256)

    Output feature shapes:
      input:  [B, 3, 32, 32]
      stage1: [B, 32, 32, 32]
      stage2: [B, 64, 16, 16]
      stage3: [B, 128, 8, 8]
      stage4: [B, 256, 4, 4]
    """
    return InceptionMambaClassifier(
        num_classes=num_classes,
        in_chans=in_chans,
        img_size_mode="32",
        depths=(1, 1, 3, 1),
        dims=(32, 64, 128, 256),
        drop_path_rate=0.05,
        mlp_ratio=4.0,
        conv_branch_ratio=1.0 / 8.0,
        bottleneck_ratio=0.5,
        scan_mode="single",
        mamba_d_state=8,
        mamba_expand_factor=1,
        mamba_d_conv=3,
    )


# Backward-compatible aliases
def inception_mamba_tiny_224(num_classes: int = 1000) -> InceptionMambaClassifier:
    return inception_mamba_tiny_224_paper(num_classes=num_classes, in_chans=3)


def inception_mamba_tiny_32(num_classes: int = 10) -> InceptionMambaClassifier:
    return inception_mamba_cifar_32(num_classes=num_classes, in_chans=3)


def count_parameters(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


# Random tests
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"mambapy available: {HAS_MAMBA_PY}")

    # 224x224 paper Tiny test
    model_224 = inception_mamba_tiny_224_paper(
        num_classes=1000,
        in_chans=3,
        scan_mode="single",
    ).to(device)

    model_224.eval()

    x224 = torch.randn(1, 3, 224, 224, device=device)

    with torch.no_grad():
        y224 = model_224(x224)
        feats224 = model_224.forward_features(x224, return_all=True)

    print("\n[224x224 Paper Tiny]")
    print(f"Mamba backend: {model_224.backbone.mamba_backend}")
    print(f"Scan mode: {model_224.backbone.scan_mode}")
    print(f"Params: {count_parameters(model_224):.2f} M")
    print(f"Input:  {tuple(x224.shape)}")
    print(f"Output: {tuple(y224.shape)}")

    for i, f in enumerate(feats224, 1):
        print(f"Stage {i}: {tuple(f.shape)}")

    assert y224.shape == (1, 1000)

    del model_224, x224, y224, feats224

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 32x32 CIFAR reduced test
    model_32 = inception_mamba_cifar_32(
        num_classes=10,
        in_chans=3,
    ).to(device)

    model_32.eval()

    x32 = torch.randn(2, 3, 32, 32, device=device)

    with torch.no_grad():
        y32 = model_32(x32)
        feats32 = model_32.forward_features(x32, return_all=True)

    print("\n[32x32 CIFAR Reduced]")
    print(f"Mamba backend: {model_32.backbone.mamba_backend}")
    print(f"Scan mode: {model_32.backbone.scan_mode}")
    print(f"Params: {count_parameters(model_32):.2f} M")
    print(f"Input:  {tuple(x32.shape)}")
    print(f"Output: {tuple(y32.shape)}")

    for i, f in enumerate(feats32, 1):
        print(f"Stage {i}: {tuple(f.shape)}")

    assert y32.shape == (2, 10)

    print("\nAll random tests passed.")