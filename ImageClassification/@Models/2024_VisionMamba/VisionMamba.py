"""
VisionMamba.py
==============

Three pure-PyTorch Vision Mamba classifiers:

    VisionMamba     : paper-style, higher-capacity Vim architecture
    VisionMamba_S   : smaller/lightweight paper-style architecture
    VisionMamba_N   : very small fast architecture for CIFAR-10/CIFAR-100

The implementation runs on native Windows 11 with ordinary PyTorch and does
not require mamba-ssm, Triton, Visual Studio Build Tools, or custom CUDA code.

Important:
    This reproduces the architecture and selective-SSM mathematics in pure
    PyTorch.  It is not binary/state-dict compatible with the official HUST
    Vim checkpoints, because the official repository uses custom fused kernels
    and a repository-specific BiMamba implementation.

Example:
    from VisionMamba import vision_mamba_n

    model = vision_mamba_n(num_classes=10)
    x = torch.randn(8, 3, 32, 32)
    logits = model(x)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn

from mamba import MambaBlock, MambaConfig, RMSNorm, count_parameters


@dataclass(frozen=True)
class VisionMambaConfig:
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    num_classes: int = 1000

    embed_dim: int = 384
    depth: int = 24
    d_state: int = 16
    d_conv: int = 4
    expand: float = 2.0
    dt_rank: Optional[int] = None

    drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    norm_eps: float = 1e-5

    # Vim inserts one class token into the middle of the patch sequence.
    use_cls_token: bool = True
    cls_token_position: str = "middle"  # "middle", "first", or "last"
    use_abs_pos_embed: bool = True
    final_pool: str = "cls"             # "cls" or "mean"
    merge_mode: str = "mean"

    def __post_init__(self) -> None:
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")
        if self.depth < 1:
            raise ValueError("depth must be at least 1.")
        if self.cls_token_position not in {"middle", "first", "last"}:
            raise ValueError(
                "cls_token_position must be 'middle', 'first', or 'last'."
            )
        if self.final_pool not in {"cls", "mean"}:
            raise ValueError("final_pool must be 'cls' or 'mean'.")
        if self.final_pool == "cls" and not self.use_cls_token:
            raise ValueError("final_pool='cls' requires use_cls_token=True.")


class PatchEmbed(nn.Module):
    """Non-overlapping image-to-patch projection."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = (
            image_size // patch_size,
            image_size // patch_size,
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected image tensor [B,C,H,W], got {x.shape}")
        if x.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                f"Expected {self.image_size}x{self.image_size} images, "
                f"got {x.shape[-2]}x{x.shape[-1]}. "
                "Resize the input or construct the model with another image_size."
            )
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class VisionMambaModel(nn.Module):
    """
    Vision Mamba (Vim-style) image classifier.

    Structure:
        image -> patch embedding
        -> middle class token + absolute positional embedding
        -> stacked bidirectional Mamba residual blocks
        -> final RMSNorm
        -> CLS or mean pooling
        -> classifier
    """

    def __init__(self, config: VisionMambaConfig) -> None:
        super().__init__()
        self.config = config
        self.num_classes = config.num_classes
        self.embed_dim = config.embed_dim

        self.patch_embed = PatchEmbed(
            image_size=config.image_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
        )
        token_count = self.patch_embed.num_patches
        if config.use_cls_token:
            token_count += 1

        if config.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        else:
            self.register_parameter("cls_token", None)

        if config.use_abs_pos_embed:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, token_count, config.embed_dim)
            )
        else:
            self.register_parameter("pos_embed", None)

        self.pos_drop = nn.Dropout(config.drop_rate)

        mamba_config = MambaConfig(
            d_model=config.embed_dim,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            dt_rank=config.dt_rank,
        )

        drop_path_rates = torch.linspace(
            0.0, config.drop_path_rate, config.depth
        ).tolist()
        self.blocks = nn.ModuleList(
            [
                MambaBlock(
                    mamba_config,
                    bidirectional=True,
                    drop_path=drop_path_rates[index],
                    norm_eps=config.norm_eps,
                    merge_mode=config.merge_mode,
                )
                for index in range(config.depth)
            ]
        )

        self.norm = RMSNorm(config.embed_dim, eps=config.norm_eps)
        self.head = (
            nn.Linear(config.embed_dim, config.num_classes)
            if config.num_classes > 0
            else nn.Identity()
        )

        self.apply(self._init_weights)
        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(
                module.weight, mode="fan_out", nonlinearity="relu"
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _insert_cls_token(self, x: Tensor) -> Tuple[Tensor, Optional[int]]:
        if self.cls_token is None:
            return x, None

        cls = self.cls_token.expand(x.shape[0], -1, -1)
        position = self.config.cls_token_position

        if position == "first":
            return torch.cat((cls, x), dim=1), 0
        if position == "last":
            return torch.cat((x, cls), dim=1), x.shape[1]

        middle = x.shape[1] // 2
        return torch.cat((x[:, :middle], cls, x[:, middle:]), dim=1), middle

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)
        x, cls_index = self._insert_cls_token(x)

        if self.pos_embed is not None:
            x = x + self.pos_embed.to(dtype=x.dtype)
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        if self.config.final_pool == "cls":
            if cls_index is None:
                raise RuntimeError("CLS pooling requested without a CLS token.")
            return x[:, cls_index]

        if cls_index is not None:
            # Mean-pool only image patches, excluding the class token.
            x = torch.cat((x[:, :cls_index], x[:, cls_index + 1 :]), dim=1)
        return x.mean(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.forward_features(x))

    def parameter_count(self) -> int:
        return count_parameters(self)


# Presets

# Higher-capacity paper-style model.  Its 24-layer hierarchy, 16x16 patches,
# absolute position embedding, middle CLS token, and bidirectional Mamba blocks
# follow the principal Vim design.  Width 384 corresponds to the commonly used
# small-capacity scale in the official model family.
VISION_MAMBA_CONFIG = VisionMambaConfig(
    image_size=224,
    patch_size=16,
    num_classes=1000,
    embed_dim=384,
    depth=24,
    d_state=16,
    d_conv=4,
    expand=2.0,
    drop_path_rate=0.1,
    use_cls_token=True,
    cls_token_position="middle",
    use_abs_pos_embed=True,
    final_pool="cls",
)

# Lightweight paper-style model for 224x224 inputs.
VISION_MAMBA_S_CONFIG = VisionMambaConfig(
    image_size=224,
    patch_size=16,
    num_classes=1000,
    embed_dim=192,
    depth=12,
    d_state=16,
    d_conv=4,
    expand=2.0,
    drop_path_rate=0.05,
    use_cls_token=True,
    cls_token_position="middle",
    use_abs_pos_embed=True,
    final_pool="cls",
)

# Very small CIFAR model.  Patch size 8 produces a 4x4 token grid from a
# 32x32 image, minimizing Python-level scan overhead.  Mean pooling is used for stable small-dataset training.
VISION_MAMBA_N_CONFIG = VisionMambaConfig(
    image_size=32,
    patch_size=8,
    num_classes=10,
    embed_dim=64,
    depth=4,
    d_state=8,
    d_conv=3,
    expand=1.5,
    drop_path_rate=0.05,
    use_cls_token=True,
    cls_token_position="middle",
    use_abs_pos_embed=True,
    final_pool="mean",
)


class VisionMamba(VisionMambaModel):
    """Paper-style higher-capacity Vision Mamba."""

    def __init__(self, num_classes: int = 1000, **overrides) -> None:
        config = replace(
            VISION_MAMBA_CONFIG,
            num_classes=num_classes,
            **overrides,
        )
        super().__init__(config)


class VisionMamba_S(VisionMambaModel):
    """Lightweight paper-style Vision Mamba."""

    def __init__(self, num_classes: int = 1000, **overrides) -> None:
        config = replace(
            VISION_MAMBA_S_CONFIG,
            num_classes=num_classes,
            **overrides,
        )
        super().__init__(config)


class VisionMamba_N(VisionMambaModel):
    """Very small, CIFAR-oriented Vision Mamba."""

    def __init__(self, num_classes: int = 10, **overrides) -> None:
        config = replace(
            VISION_MAMBA_N_CONFIG,
            num_classes=num_classes,
            **overrides,
        )
        super().__init__(config)


def vision_mamba(num_classes: int = 1000, **kwargs) -> VisionMamba:
    return VisionMamba(num_classes=num_classes, **kwargs)


def vision_mamba_s(num_classes: int = 1000, **kwargs) -> VisionMamba_S:
    return VisionMamba_S(num_classes=num_classes, **kwargs)


def vision_mamba_n(num_classes: int = 10, **kwargs) -> VisionMamba_N:
    return VisionMamba_N(num_classes=num_classes, **kwargs)


MODEL_REGISTRY: Dict[str, type[VisionMambaModel]] = {
    "VisionMamba": VisionMamba,
    "VisionMamba-S": VisionMamba_S,
    "VisionMamba-N": VisionMamba_N,
}


def create_model(name: str, num_classes: Optional[int] = None, **kwargs):
    """Construct a preset by name."""
    if name not in MODEL_REGISTRY:
        choices = ", ".join(MODEL_REGISTRY)
        raise KeyError(f"Unknown model {name!r}. Available models: {choices}")

    model_cls = MODEL_REGISTRY[name]
    if num_classes is not None:
        kwargs["num_classes"] = num_classes
    return model_cls(**kwargs)


if __name__ == "__main__":
    # Fast native-Windows smoke test.
    torch.manual_seed(0)
    model = VisionMamba_N(num_classes=10)
    sample = torch.randn(1, 3, 32, 32)

    model.eval()
    with torch.inference_mode():
        logits = model(sample)

    print(model.__class__.__name__)
    print(f"parameters: {model.parameter_count():,}")
    print(f"input:      {tuple(sample.shape)}")
    print(f"output:     {tuple(logits.shape)}")
