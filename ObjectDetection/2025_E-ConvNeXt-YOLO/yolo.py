"""
yolo.py

E-ConvNeXt-mini/small + YOLOv10n-style objectness-free detector for Pascal VOC.

Design goals:
  * Keep the E-ConvNeXt paper's CSP ConvNeXt block: stepped stem, BN, 7x7 DWConv,
    1x1 MLP, ESE channel attention, and CSP stage ratio.
  * Replace the old detector's RGB/IINet-specific backbone with an E-ConvNeXt backbone.
  * Keep a YOLOv10-like, anchor-free, objectness-free dense head with dual one-to-many
    and one-to-one branches. Training uses both branches; inference defaults to one-to-one.
  * Output order is [large, medium, small] so it remains compatible with your old loss/eval style.

Tensor format per detection level:
  (B, 4 + num_classes, H, W), channels [tx, ty, tw, th, class_logits...]
  Decode:
      x = (sigmoid(tx) + grid_x) / W
      y = (sigmoid(ty) + grid_y) / H
      w = exp(clamp(tw)) / W
      h = exp(clamp(th)) / H
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "make_divisible",
    "DropPath",
    "ESEBlock",
    "EConvNeXtBlock",
    "CSPConvNeXtStage",
    "EConvNeXtBackbone",
    "YOLOv10EConvNeXt",
    "build_yolov10_econvnext",
    "count_parameters",
]


def make_divisible(v: float, divisor: int = 8) -> int:
    return max(divisor, int((v + divisor / 2) // divisor * divisor))


class DropPath(nn.Module):
    """Stochastic depth. Disabled automatically during eval."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x.div(keep) * mask


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        act: Union[bool, str] = True,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        if act is True:
            self.act = nn.SiLU(inplace=False)
        elif act == "gelu":
            self.act = nn.GELU()
        elif act == "silu":
            self.act = nn.SiLU(inplace=False)
        else:
            self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DWConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, act: Union[bool, str] = True):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(in_channels, in_channels, 3, stride=stride, groups=in_channels, act=act),
            ConvBNAct(in_channels, out_channels, 1, padding=0, act=act),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ESEBlock(nn.Module):
    """Effective Squeeze-and-Excitation: GAP -> 1x1 conv -> sigmoid."""

    def __init__(self, channels: int):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.act(self.fc(self.avgpool(x)))


class EConvNeXtBlock(nn.Module):
    """E-ConvNeXt block: 7x7 DWConv + BN + 1x1 MLP + ESE + residual.

    The last BN is zero-initialized to make the residual branch start as identity,
    which is helpful when training detection from scratch on VOC.
    """

    def __init__(self, channels: int, mlp_ratio: float = 4.0, drop_path: float = 0.0):
        super().__init__()
        hidden_channels = make_divisible(channels * mlp_ratio)
        self.dwconv = nn.Conv2d(channels, channels, 7, stride=1, padding=3, groups=channels, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.pwconv1 = nn.Conv2d(channels, hidden_channels, 1, bias=False)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(hidden_channels, channels, 1, bias=False)
        self.ese = ESEBlock(channels)
        self.out_bn = nn.BatchNorm2d(channels)
        self.drop = DropPath(drop_path)
        nn.init.constant_(self.out_bn.weight, 0.0)
        nn.init.constant_(self.out_bn.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        y = self.dwconv(x)
        y = self.bn(y)
        y = self.pwconv1(y)
        y = self.act(y)
        y = self.pwconv2(y)
        y = self.ese(y)
        y = self.out_bn(y)
        return identity + self.drop(y)


class CSPConvNeXtStage(nn.Module):
    """CSP E-ConvNeXt stage.

    It uses a 3x3 stride-2 downsample, 1x1 conv split, ConvNeXt blocks on the
    transformed branch, a shortcut branch, then a 1x1 merge.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        stage_ratio: float = 0.5,
        drop_path_rates: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        self.downsample = ConvBNAct(in_channels, out_channels, 3, stride=2, padding=1, act="gelu")
        block_channels = max(8, int(out_channels * stage_ratio))
        shortcut_channels = out_channels - block_channels
        if shortcut_channels <= 0:
            raise ValueError("stage_ratio must leave at least one shortcut channel")
        self.part1_conv = ConvBNAct(out_channels, block_channels, 1, padding=0, act="gelu")
        self.part2_conv = ConvBNAct(out_channels, shortcut_channels, 1, padding=0, act="gelu")
        if drop_path_rates is None:
            drop_path_rates = [0.0] * depth
        self.blocks = nn.Sequential(
            *[EConvNeXtBlock(block_channels, drop_path=float(drop_path_rates[i])) for i in range(depth)]
        )
        self.merge_conv = ConvBNAct(out_channels, out_channels, 1, padding=0, act="gelu")
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x1 = self.part1_conv(x)
        x2 = self.part2_conv(x)
        x1 = self.blocks(x1)
        return self.merge_conv(torch.cat([x1, x2], dim=1))


class SteppedStem(nn.Module):
    """E-ConvNeXt stepped stem: one stride-2 conv then two stride-1 convs."""

    def __init__(self, img_channels: int, out_channels: int):
        super().__init__()
        mid = max(8, out_channels // 2)
        self.stem = nn.Sequential(
            ConvBNAct(img_channels, mid, 3, stride=2, padding=1, act="gelu"),
            ConvBNAct(mid, mid, 3, stride=1, padding=1, act="gelu"),
            ConvBNAct(mid, out_channels, 3, stride=1, padding=1, act="gelu"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class EConvNeXtBackbone(nn.Module):
    """Backbone returning C2/C3/C4/C5 with strides 4/8/16/32."""

    ARCHS = {
        "mini": dict(depths=(2, 2, 6, 2), dims=(48, 96, 192, 384), stage_ratio=0.5),
        "tiny": dict(depths=(3, 3, 9, 3), dims=(64, 128, 256, 512), stage_ratio=0.5),
        "small": dict(depths=(3, 3, 15, 3), dims=(64, 128, 256, 512), stage_ratio=0.5),
    }

    def __init__(
        self,
        variant: str = "mini",
        img_channels: int = 3,
        stage_ratio: Optional[float] = None,
        drop_path_rate: float = 0.05,
    ):
        super().__init__()
        variant = variant.lower()
        if variant not in self.ARCHS:
            raise ValueError(f"variant must be one of {sorted(self.ARCHS)}, got {variant!r}")
        cfg = self.ARCHS[variant]
        depths = list(cfg["depths"])
        dims = list(cfg["dims"])
        ratio = float(cfg["stage_ratio"] if stage_ratio is None else stage_ratio)
        self.variant = variant
        self.depths = depths
        self.dims = dims
        self.out_channels = tuple(dims)

        total_blocks = sum(depths)
        dpr = torch.linspace(0, drop_path_rate, total_blocks).tolist() if total_blocks > 0 else []
        idx = 0

        self.stem = SteppedStem(img_channels, dims[0])
        self.stage1 = CSPConvNeXtStage(dims[0], dims[0], depths[0], ratio, dpr[idx : idx + depths[0]])
        idx += depths[0]
        self.stage2 = CSPConvNeXtStage(dims[0], dims[1], depths[1], ratio, dpr[idx : idx + depths[1]])
        idx += depths[1]
        self.stage3 = CSPConvNeXtStage(dims[1], dims[2], depths[2], ratio, dpr[idx : idx + depths[2]])
        idx += depths[2]
        self.stage4 = CSPConvNeXtStage(dims[2], dims[3], depths[3], ratio, dpr[idx : idx + depths[3]])
        self.apply(self._init_weights)
        # Preserve residual identity start after global initialization.
        for module in self.modules():
            if isinstance(module, EConvNeXtBlock):
                nn.init.constant_(module.out_bn.weight, 0.0)
                nn.init.constant_(module.out_bn.bias, 0.0)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)      # stride 2
        c2 = self.stage1(x)   # stride 4
        c3 = self.stage2(c2)  # stride 8
        c4 = self.stage3(c3)  # stride 16
        c5 = self.stage4(c4)  # stride 32
        return c2, c3, c4, c5


class BottleneckLite(nn.Module):
    def __init__(self, channels: int, expansion: float = 0.5, shortcut: bool = True, drop_path: float = 0.0):
        super().__init__()
        hidden = make_divisible(channels * expansion)
        self.cv1 = ConvBNAct(channels, hidden, 1, padding=0, act="silu")
        self.cv2 = DWConvBNAct(hidden, channels, stride=1, act="silu")
        self.add = shortcut
        self.drop = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + self.drop(y) if self.add else y


class C2fLite(nn.Module):
    """YOLO-style C2f block made lighter with depthwise bottlenecks."""

    def __init__(self, in_channels: int, out_channels: int, n: int = 2, expansion: float = 0.5, drop_path: float = 0.0):
        super().__init__()
        hidden = make_divisible(out_channels * expansion)
        self.cv1 = ConvBNAct(in_channels, hidden * 2, 1, padding=0, act="silu")
        self.blocks = nn.ModuleList([BottleneckLite(hidden, expansion=1.0, drop_path=drop_path) for _ in range(n)])
        self.cv2 = ConvBNAct(hidden * (2 + n), out_channels, 1, padding=0, act="silu")
        self.ese = ESEBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, dim=1))
        for block in self.blocks:
            y.append(block(y[-1]))
        return self.ese(self.cv2(torch.cat(y, dim=1)))


class SPPF(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool_size: int = 5):
        super().__init__()
        hidden = make_divisible(in_channels // 2)
        self.cv1 = ConvBNAct(in_channels, hidden, 1, padding=0, act="silu")
        self.pool = nn.MaxPool2d(pool_size, stride=1, padding=pool_size // 2)
        self.cv2 = ConvBNAct(hidden * 4, out_channels, 1, padding=0, act="silu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


class YOLOv10NanoPAFPN(nn.Module):
    """YOLOv10n-style PAFPN neck over E-ConvNeXt C3/C4/C5 features."""

    def __init__(
        self,
        in_channels: Sequence[int],
        neck_channels: int = 96,
        depth_mult: float = 1.0,
        drop_path: float = 0.02,
    ):
        super().__init__()
        c3_in, c4_in, c5_in = in_channels
        c3 = make_divisible(neck_channels)
        c4 = make_divisible(neck_channels * 2)
        c5 = make_divisible(neck_channels * 4)
        self.out_channels = (c5, c4, c3)  # large, medium, small
        n = max(1, round(2 * depth_mult))

        self.proj_c3 = ConvBNAct(c3_in, c3, 1, padding=0, act="silu")
        self.proj_c4 = ConvBNAct(c4_in, c4, 1, padding=0, act="silu")
        self.proj_c5 = ConvBNAct(c5_in, c5, 1, padding=0, act="silu")
        self.sppf = SPPF(c5, c5)

        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.fpn_p4 = C2fLite(c5 + c4, c4, n=n, drop_path=drop_path)
        self.fpn_p3 = C2fLite(c4 + c3, c3, n=n, drop_path=drop_path)
        self.down_p3 = DWConvBNAct(c3, c3, stride=2, act="silu")
        self.pan_p4 = C2fLite(c3 + c4, c4, n=n, drop_path=drop_path)
        self.down_p4 = DWConvBNAct(c4, c5, stride=2, act="silu")
        self.pan_p5 = C2fLite(c5 + c5, c5, n=n, drop_path=drop_path)

    def forward(self, c3: torch.Tensor, c4: torch.Tensor, c5: torch.Tensor) -> List[torch.Tensor]:
        p3 = self.proj_c3(c3)
        p4 = self.proj_c4(c4)
        p5 = self.sppf(self.proj_c5(c5))

        f4 = self.fpn_p4(torch.cat([self.up(p5), p4], dim=1))
        f3 = self.fpn_p3(torch.cat([self.up(f4), p3], dim=1))
        n4 = self.pan_p4(torch.cat([self.down_p3(f3), f4], dim=1))
        n5 = self.pan_p5(torch.cat([self.down_p4(n4), p5], dim=1))
        return [n5, n4, f3]  # large, medium, small


class DecoupledDetectBlock(nn.Module):
    """Objectness-free decoupled head for one level."""

    def __init__(self, in_channels: int, num_classes: int, hidden_ratio: float = 0.75):
        super().__init__()
        hidden = max(make_divisible(in_channels * hidden_ratio), 64)
        self.stem = ConvBNAct(in_channels, hidden, 1, padding=0, act="silu")
        self.box_tower = nn.Sequential(
            DWConvBNAct(hidden, hidden, stride=1, act="silu"),
            DWConvBNAct(hidden, hidden, stride=1, act="silu"),
        )
        self.cls_tower = nn.Sequential(
            DWConvBNAct(hidden, hidden, stride=1, act="silu"),
            DWConvBNAct(hidden, hidden, stride=1, act="silu"),
        )
        self.box_pred = nn.Conv2d(hidden, 4, 1)
        self.cls_pred = nn.Conv2d(hidden, num_classes, 1)
        self._init_bias(num_classes)

    def _init_bias(self, num_classes: int) -> None:
        nn.init.constant_(self.box_pred.bias, 0.0)
        prior = 0.01
        bias = -math.log((1.0 - prior) / prior)
        nn.init.constant_(self.cls_pred.bias, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        box_feat = self.box_tower(x)
        cls_feat = self.cls_tower(x)
        return torch.cat([self.box_pred(box_feat), self.cls_pred(cls_feat)], dim=1)


class YOLOv10DualHead(nn.Module):
    """Dual assignment head: one-to-many branch + one-to-one branch."""

    def __init__(self, in_channels: Sequence[int], num_classes: int, detach_one2one: bool = True):
        super().__init__()
        self.num_classes = int(num_classes)
        self.detach_one2one = bool(detach_one2one)
        self.one2many = nn.ModuleList([DecoupledDetectBlock(c, num_classes) for c in in_channels])
        self.one2one = nn.ModuleList([DecoupledDetectBlock(c, num_classes) for c in in_channels])

    def forward(self, feats: Sequence[torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        one2many = [head(x) for head, x in zip(self.one2many, feats)]
        if self.detach_one2one:
            one2one = [head(x.detach()) for head, x in zip(self.one2one, feats)]
        else:
            one2one = [head(x) for head, x in zip(self.one2one, feats)]
        return {"one2many": one2many, "one2one": one2one}


class YOLOv10EConvNeXt(nn.Module):
    """E-ConvNeXt backbone + YOLOv10n-style neck/head."""

    def __init__(
        self,
        num_classes: int = 20,
        variant: str = "mini",
        img_channels: int = 3,
        neck_channels: Optional[int] = None,
        depth_mult: Optional[float] = None,
        drop_path_rate: float = 0.05,
        detach_one2one: bool = True,
    ):
        super().__init__()
        variant = variant.lower()
        self.num_classes = int(num_classes)
        self.variant = variant
        self.backbone = EConvNeXtBackbone(variant=variant, img_channels=img_channels, drop_path_rate=drop_path_rate)
        dims = self.backbone.out_channels
        # VOC on 8 GB cards: mini uses 96; small uses 128 by default.
        if neck_channels is None:
            neck_channels = 96 if variant == "mini" else 128
        if depth_mult is None:
            depth_mult = 0.67 if variant == "mini" else 1.0
        self.neck = YOLOv10NanoPAFPN(dims[1:4], neck_channels=neck_channels, depth_mult=depth_mult)
        self.head = YOLOv10DualHead(self.neck.out_channels, num_classes=num_classes, detach_one2one=detach_one2one)

    def forward(
        self,
        x: torch.Tensor,
        branch: str = "auto",
    ) -> Union[Dict[str, List[torch.Tensor]], List[torch.Tensor]]:
        _, c3, c4, c5 = self.backbone(x)
        feats = self.neck(c3, c4, c5)
        out = self.head(feats)
        if branch == "auto":
            return out if self.training else out["one2one"]
        if branch in out:
            return out[branch]
        if branch == "both":
            return out
        raise ValueError("branch must be auto, both, one2many, or one2one")

    def load_backbone_weights(self, checkpoint_path: str, strict: bool = False) -> Tuple[List[str], List[str]]:
        """Load classifier-style E-ConvNeXt weights into the backbone when keys match."""
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(ckpt, dict):
            state = ckpt.get("model", ckpt.get("state_dict", ckpt))
        else:
            state = ckpt
        cleaned = {}
        for k, v in state.items():
            k2 = str(k)
            if k2.startswith("module."):
                k2 = k2[7:]
            if k2.startswith("backbone."):
                k2 = k2[len("backbone.") :]
            if k2.startswith("head.") or k2.startswith("avgpool"):
                continue
            cleaned[k2] = v
        msg = self.backbone.load_state_dict(cleaned, strict=strict)
        return list(msg.missing_keys), list(msg.unexpected_keys)


def build_yolov10_econvnext(
    num_classes: int = 20,
    variant: str = "mini",
    img_channels: int = 3,
    **kwargs,
) -> YOLOv10EConvNeXt:
    return YOLOv10EConvNeXt(num_classes=num_classes, variant=variant, img_channels=img_channels, **kwargs)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> float:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    return sum(p.numel() for p in model.parameters()) / 1e6


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_yolov10_econvnext(num_classes=20, variant="mini").to(device).eval()
    x = torch.randn(2, 3, 512, 512, device=device)
    with torch.no_grad():
        y = model(x, branch="one2one")
    print("params(M):", count_parameters(model, trainable_only=False))
    print([tuple(t.shape) for t in y])
