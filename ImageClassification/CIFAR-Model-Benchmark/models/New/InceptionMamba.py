# InceptionMamba.py
# Windows + PyTorch compatible
#
# Drop-in model file for your framework.
#
# This version keeps your original style:
#   - build_InceptionMamba_Tiny(num_classes, img_channels)
#   - inception_mamba_tiny_224(...)
#   - count_parameters(...)
#
# Main fixes compared with your current file:
#   1. BottleneckMamba has inner residual:
#        y = x + scale(expand(SS2D(reduce(x))))
#   2. ConvMixer uses paper-style large band conv:
#        3x3 + 3x11/11x3 + identity
#   3. Stem/head are closer to official implementation style.
#   4. FallbackSS2D is trainable and stable even without official selective_scan.
#
# Optional:
#   pip install mambapy
#
# Note:
#   The real paper/official implementation uses SS2D selective scan.
#   This file provides a stable fallback SS2D-like mixer so training can converge
#   before you set up official CUDA kernels.

from __future__ import annotations

from typing import List, Tuple, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    For exact paper-style spatial scanning, use scan_mode="cross4".
    If your GPU is small, change to scan_mode="single".
    """
    return inception_mamba_tiny_224(
        num_classes=num_classes,
        in_chans=img_channels,
        scan_mode="cross4",
        mamba_backend="fallback",   # "fallback" is stable; use "mambapy" only if installed and tested
    )


# Basic layers

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


class LayerNorm2d(nn.Module):
    """
    Channel-first LayerNorm for NCHW tensors.
    Kept for optional use, but this model mainly uses BatchNorm2d.
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


class LayerScale2d(nn.Module):
    """
    Per-channel layer scale for NCHW.
    """
    def __init__(self, dim: int, init_value: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma[:, None, None]


# ConvMixer: 3x3 + 3x11/11x3 + identity

class InceptionBandConvMixer(nn.Module):
    """
    Paper-style ConvMixer.

    Input channels are split into:
      square branch:   1/8 C -> depthwise 3x3
      band branch:     1/8 C -> depthwise 3x11 + depthwise 11x3
      identity branch: remaining channels

    Output:
      concat(square, band, identity)
    """
    def __init__(
        self,
        dim: int,
        square_kernel: int = 3,
        band_kernel: int = 11,
        branch_ratio: float = 1.0 / 8.0,
    ):
        super().__init__()

        square_ch = int(dim * branch_ratio)
        band_ch = int(dim * branch_ratio)

        square_ch = max(1, square_ch)
        band_ch = max(1, band_ch)

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


# Sequence mixers

class FallbackSequenceMixer(nn.Module):
    """
    Stable trainable fallback sequence mixer.

    This is NOT exact Mamba.
    It is used only when official SS2D/selective_scan is unavailable.

    Input:
      x: [B, L, C]

    Output:
      y: [B, L, C]
    """
    def __init__(
        self,
        dim: int,
        d_conv: int = 7,
        expand_ratio: float = 2.0,
    ):
        super().__init__()

        hidden_dim = int(dim * expand_ratio)

        self.norm = nn.LayerNorm(dim)

        self.in_proj = nn.Linear(dim, hidden_dim * 2)
        self.dwconv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=d_conv,
            padding=d_conv // 2,
            groups=hidden_dim,
            bias=True,
        )
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)

        u, gate = self.in_proj(z).chunk(2, dim=-1)

        u = u.transpose(1, 2).contiguous()
        u = self.dwconv(u)
        u = u.transpose(1, 2).contiguous()

        u = self.act(u)
        y = u * torch.sigmoid(gate)

        y = self.out_proj(y)

        return y


class MambaPySequenceMixer(nn.Module):
    """
    Optional mambapy wrapper.

    Input:
      x: [B, L, C]

    Output:
      y: [B, L, C]
    """
    def __init__(
        self,
        dim: int,
        d_state: int = 8,
        expand_factor: int = 1,
        d_conv: int = 3,
    ):
        super().__init__()

        if not HAS_MAMBA_PY:
            raise RuntimeError("mambapy is not installed. Please use mamba_backend='fallback'.")

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mixer(x)


# SS2D-like spatial mixer

class SpatialSS2DLike(nn.Module):
    """
    SS2D-like spatial scan.

    scan_mode:
      "single":
          row-major only

      "cross4":
          row-major
          reverse row-major
          column-major
          reverse column-major

    This class can use:
      - fallback sequence mixer
      - mambapy sequence mixer

    The paper/official version uses selective_scan SS2D.
    This wrapper preserves the 2D cross-scan behavior and model topology.
    """
    def __init__(
        self,
        dim: int,
        scan_mode: Literal["single", "cross4"] = "cross4",
        mamba_backend: Literal["fallback", "mambapy"] = "fallback",
        mamba_d_state: int = 8,
        mamba_expand_factor: int = 1,
        mamba_d_conv: int = 3,
    ):
        super().__init__()

        if scan_mode not in ("single", "cross4"):
            raise ValueError("scan_mode must be 'single' or 'cross4'.")

        if mamba_backend not in ("fallback", "mambapy"):
            raise ValueError("mamba_backend must be 'fallback' or 'mambapy'.")

        self.scan_mode = scan_mode
        self.mamba_backend = mamba_backend

        if mamba_backend == "mambapy":
            self.seq_mixer = MambaPySequenceMixer(
                dim=dim,
                d_state=mamba_d_state,
                expand_factor=mamba_expand_factor,
                d_conv=mamba_d_conv,
            )
        else:
            self.seq_mixer = FallbackSequenceMixer(
                dim=dim,
                d_conv=7,
                expand_ratio=2.0,
            )

    def _scan_hw(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W] -> [B, H*W, C] -> [B, C, H, W]
        b, c, h, w = x.shape

        seq = x.flatten(2).transpose(1, 2).contiguous()
        out = self.seq_mixer(seq)
        out = out.transpose(1, 2).contiguous().reshape(b, c, h, w)

        return out

    def _scan_wh(self, x: torch.Tensor) -> torch.Tensor:
        # column-major scan by transposing H/W
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


# Bottleneck Mamba GlobalMixer

class BottleneckMambaGlobalMixer(nn.Module):
    """
    Paper-style GlobalMixer with bottleneck.

    Important:
      The inner residual is required.

    Structure:
      y = Conv1x1(C -> C/2)
      y = SS2D(y)
      y = scale(y)
      y = Conv1x1(C/2 -> C)
      out = x + y

    This is the main thing missing in your original code.
    """
    def __init__(
        self,
        dim: int,
        bottleneck_ratio: float = 0.5,
        scan_mode: Literal["single", "cross4"] = "cross4",
        mamba_backend: Literal["fallback", "mambapy"] = "fallback",
        mamba_d_state: int = 8,
        mamba_expand_factor: int = 1,
        mamba_d_conv: int = 3,
        base_scale_init_value: float = 1e-6,
    ):
        super().__init__()

        hidden_dim = max(1, int(dim * bottleneck_ratio))

        self.reduce = nn.Conv2d(
            dim,
            hidden_dim,
            kernel_size=1,
            bias=True,
        )

        self.ss2d = SpatialSS2DLike(
            dim=hidden_dim,
            scan_mode=scan_mode,
            mamba_backend=mamba_backend,
            mamba_d_state=mamba_d_state,
            mamba_expand_factor=mamba_expand_factor,
            mamba_d_conv=mamba_d_conv,
        )

        self.base_scale = LayerScale2d(
            hidden_dim,
            init_value=base_scale_init_value,
        )

        self.expand = nn.Conv2d(
            hidden_dim,
            dim,
            kernel_size=1,
            bias=True,
        )

    @property
    def backend(self) -> str:
        return self.ss2d.mamba_backend

    @property
    def scan_mode(self) -> str:
        return self.ss2d.scan_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        y = self.reduce(x)
        y = self.ss2d(y)
        y = self.base_scale(y)
        y = self.expand(y)

        return shortcut + y


# InceptionMamba block

class InceptionMambaBlock(nn.Module):
    """
    InceptionMamba block.

    Paper diagram:
      x
      -> ConvMixer
      -> GlobalMixer / Bottleneck Mamba
      -> Norm
      -> MLP
      -> LayerScale
      -> DropPath
      -> residual add
    """
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        conv_branch_ratio: float = 1.0 / 8.0,
        bottleneck_ratio: float = 0.5,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        base_scale_init_value: float = 1e-6,
        scan_mode: Literal["single", "cross4"] = "cross4",
        mamba_backend: Literal["fallback", "mambapy"] = "fallback",
        mamba_d_state: int = 8,
        mamba_expand_factor: int = 1,
        mamba_d_conv: int = 3,
        norm_layer: str = "bn",
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
            mamba_backend=mamba_backend,
            mamba_d_state=mamba_d_state,
            mamba_expand_factor=mamba_expand_factor,
            mamba_d_conv=mamba_d_conv,
            base_scale_init_value=base_scale_init_value,
        )

        if norm_layer == "bn":
            self.norm = nn.BatchNorm2d(dim)
        elif norm_layer == "ln":
            self.norm = LayerNorm2d(dim)
        else:
            raise ValueError("norm_layer must be 'bn' or 'ln'.")

        hidden_dim = int(dim * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=True),
        )

        self.layer_scale = LayerScale2d(
            dim,
            init_value=layer_scale_init_value,
        )

        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        x = self.conv_mixer(x)
        x = self.global_mixer(x)
        x = self.norm(x)
        x = self.mlp(x)
        x = self.layer_scale(x)

        return shortcut + self.drop_path(x)


# Stem / Downsample

class PatchEmbed224(nn.Module):
    """
    224x224 stem.

    Produces stage-1 feature map:
      [B, C, H/4, W/4]

    This is closer to official Conv-BN-GELU stem than your original LayerNorm stem.
    """
    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
    ):
        super().__init__()

        mid_dim = embed_dim // 2

        self.proj = nn.Sequential(
            nn.Conv2d(
                in_chans,
                mid_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(mid_dim),
            nn.GELU(),

            nn.Conv2d(
                mid_dim,
                embed_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Downsample(nn.Module):
    """
    Stage downsampling:
      BN -> Conv3x3 stride 2
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
    ):
        super().__init__()

        self.down = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.Conv2d(
                in_dim,
                out_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


# Backbone

class InceptionMambaBackbone(nn.Module):
    """
    InceptionMamba backbone.

    Tiny 224x224 default:
      depths = (3, 3, 12, 3)
      dims   = (72, 144, 288, 576)

    For 224x224 input:
      stage1: [B, 72,  56, 56]
      stage2: [B, 144, 28, 28]
      stage3: [B, 288, 14, 14]
      stage4: [B, 576,  7,  7]
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
        mamba_backend: Literal["fallback", "mambapy"] = "fallback",
        mamba_d_state: int = 8,
        mamba_expand_factor: int = 1,
        mamba_d_conv: int = 3,
        layer_scale_init_value: float = 1e-6,
        base_scale_init_value: float = 1e-6,
        norm_layer: str = "bn",
    ):
        super().__init__()

        self.depths = depths
        self.dims = dims
        self.out_channels = dims

        self.patch_embed = PatchEmbed224(
            in_chans=in_chans,
            embed_dim=dims[0],
        )

        total_blocks = sum(depths)
        dp_rates = torch.linspace(0, drop_path_rate, total_blocks).tolist()

        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        cur = 0

        for stage_idx in range(4):
            blocks = []

            for _ in range(depths[stage_idx]):
                blocks.append(
                    InceptionMambaBlock(
                        dim=dims[stage_idx],
                        mlp_ratio=mlp_ratio,
                        conv_branch_ratio=conv_branch_ratio,
                        bottleneck_ratio=bottleneck_ratio,
                        drop_path=dp_rates[cur],
                        layer_scale_init_value=layer_scale_init_value,
                        base_scale_init_value=base_scale_init_value,
                        scan_mode=scan_mode,
                        mamba_backend=mamba_backend,
                        mamba_d_state=mamba_d_state,
                        mamba_expand_factor=mamba_expand_factor,
                        mamba_d_conv=mamba_d_conv,
                        norm_layer=norm_layer,
                    )
                )
                cur += 1

            self.stages.append(nn.Sequential(*blocks))

            if stage_idx < 3:
                self.downsamples.append(
                    Downsample(
                        in_dim=dims[stage_idx],
                        out_dim=dims[stage_idx + 1],
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

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
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

        for stage_idx in range(4):
            x = self.stages[stage_idx](x)
            features.append(x)

            if stage_idx < 3:
                x = self.downsamples[stage_idx](x)

        if return_all:
            return features

        return x

    def forward(self, x: torch.Tensor):
        return self.forward_features(x, return_all=True)


# Classifier

class InceptionMambaClassifier(nn.Module):
    """
    InceptionMamba classifier.

    Head:
      BN -> global average pooling -> Linear -> GELU -> Linear

    For Imagenette:
      num_classes = 10
    """
    def __init__(
        self,
        num_classes: int = 1000,
        in_chans: int = 3,
        depths: Tuple[int, int, int, int] = (3, 3, 12, 3),
        dims: Tuple[int, int, int, int] = (72, 144, 288, 576),
        drop_path_rate: float = 0.1,
        scan_mode: Literal["single", "cross4"] = "cross4",
        mamba_backend: Literal["fallback", "mambapy"] = "fallback",
        norm_layer: str = "bn",
        layer_scale_init_value: float = 1e-6,
        base_scale_init_value: float = 1e-6,
    ):
        super().__init__()

        self.backbone = InceptionMambaBackbone(
            in_chans=in_chans,
            depths=depths,
            dims=dims,
            drop_path_rate=drop_path_rate,
            mlp_ratio=4.0,
            conv_branch_ratio=1.0 / 8.0,
            bottleneck_ratio=0.5,
            scan_mode=scan_mode,
            mamba_backend=mamba_backend,
            mamba_d_state=8,
            mamba_expand_factor=1,
            mamba_d_conv=3,
            layer_scale_init_value=layer_scale_init_value,
            base_scale_init_value=base_scale_init_value,
            norm_layer=norm_layer,
        )

        final_dim = dims[-1]

        self.head_norm = nn.BatchNorm2d(final_dim)

        self.head = nn.Sequential(
            nn.Linear(final_dim, final_dim),
            nn.GELU(),
            nn.Linear(final_dim, num_classes),
        )

        self.apply(self._init_classifier_weights)

    @staticmethod
    def _init_classifier_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward_features(
        self,
        x: torch.Tensor,
        return_all: bool = False,
    ):
        feats = self.backbone.forward_features(x, return_all=True)

        if return_all:
            return feats

        x = feats[-1]
        x = self.head_norm(x)
        x = x.mean(dim=(2, 3))

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x, return_all=False)
        x = self.head(x)
        return x


# Factories

def inception_mamba_tiny_224(
    num_classes: int = 1000,
    in_chans: int = 3,
    scan_mode: Literal["single", "cross4"] = "cross4",
    mamba_backend: Literal["fallback", "mambapy"] = "fallback",
    drop_path_rate: float = 0.1,
) -> InceptionMambaClassifier:
    """
    InceptionMamba-Tiny.
    Paper config:
      depths = [3, 3, 12, 3]
      dims   = [72, 144, 288, 576]
    """
    return InceptionMambaClassifier(
        num_classes=num_classes,
        in_chans=in_chans,
        depths=(3, 3, 12, 3),
        dims=(72, 144, 288, 576),
        drop_path_rate=drop_path_rate,
        scan_mode=scan_mode,
        mamba_backend=mamba_backend,
        norm_layer="bn",
        layer_scale_init_value=1e-6,
        base_scale_init_value=1e-6,
    )


def inception_mamba_small_224(
    num_classes: int = 1000,
    in_chans: int = 3,
    scan_mode: Literal["single", "cross4"] = "cross4",
    mamba_backend: Literal["fallback", "mambapy"] = "fallback",
    drop_path_rate: float = 0.2,
) -> InceptionMambaClassifier:
    """
    InceptionMamba-Small.
    Paper config:
      depths = [4, 4, 32, 4]
      dims   = [96, 192, 384, 768]
    """
    return InceptionMambaClassifier(
        num_classes=num_classes,
        in_chans=in_chans,
        depths=(4, 4, 32, 4),
        dims=(96, 192, 384, 768),
        drop_path_rate=drop_path_rate,
        scan_mode=scan_mode,
        mamba_backend=mamba_backend,
        norm_layer="bn",
        layer_scale_init_value=1e-6,
        base_scale_init_value=1e-6,
    )


def inception_mamba_base_224(
    num_classes: int = 1000,
    in_chans: int = 3,
    scan_mode: Literal["single", "cross4"] = "cross4",
    mamba_backend: Literal["fallback", "mambapy"] = "fallback",
    drop_path_rate: float = 0.3,
) -> InceptionMambaClassifier:
    """
    InceptionMamba-Base.
    Paper config:
      depths = [4, 4, 34, 4]
      dims   = [96, 192, 384, 768]
    """
    return InceptionMambaClassifier(
        num_classes=num_classes,
        in_chans=in_chans,
        depths=(4, 4, 34, 4),
        dims=(96, 192, 384, 768),
        drop_path_rate=drop_path_rate,
        scan_mode=scan_mode,
        mamba_backend=mamba_backend,
        norm_layer="bn",
        layer_scale_init_value=1e-6,
        base_scale_init_value=1e-6,
    )


def count_parameters(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


# Random test

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"mambapy available: {HAS_MAMBA_PY}")

    model = inception_mamba_tiny_224(
        num_classes=10,
        in_chans=3,
        scan_mode="cross4",
        mamba_backend="fallback",
    ).to(device)

    model.eval()

    x = torch.randn(2, 3, 224, 224, device=device)

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

    assert y.shape == (2, 10)

    print("\nRandom test passed.")