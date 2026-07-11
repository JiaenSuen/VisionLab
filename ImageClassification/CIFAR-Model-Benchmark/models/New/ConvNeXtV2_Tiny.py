import torch
import torch.nn as nn
import torch.nn.functional as F


def build_ConvNeXtV2_Tiny(num_classes, img_channels=3):
    return ConvNeXtV2(
        in_chans=img_channels,
        num_classes=num_classes,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.1
    )


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)

        random_tensor = keep_prob + torch.rand(
            shape,
            dtype=x.dtype,
            device=x.device
        )
        random_tensor.floor_()

        return x.div(keep_prob) * random_tensor


class LayerNorm(nn.Module):
    """
    ConvNeXt / ConvNeXtV2 style LayerNorm.

    data_format:
        channels_last:  [B, H, W, C]
        channels_first: [B, C, H, W]
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x,
                self.normalized_shape,
                self.weight,
                self.bias,
                self.eps
            )

        elif self.data_format == "channels_first":
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)

            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]

            return x

        else:
            raise ValueError("data_format must be channels_last or channels_first")


class GRN(nn.Module):
    """
    Global Response Normalization.

    ConvNeXtV2 的核心改動之一。
    Input shape: [B, H, W, C]
    """
    def __init__(self, dim):
        super().__init__()

        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)

        return self.gamma * (x * nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    """
    ConvNeXtV2 Block.

    Structure:
        x
        -> DWConv 7x7
        -> LayerNorm
        -> Linear 4C
        -> GELU
        -> GRN
        -> Linear C
        -> DropPath
        -> Residual Add
    """
    def __init__(self, dim, drop_path=0.0):
        super().__init__()

        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=7,
            padding=3,
            groups=dim
        )

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")

        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        identity = x

        x = self.dwconv(x)

        # [B, C, H, W] -> [B, H, W, C]
        x = x.permute(0, 2, 3, 1)

        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)

        # [B, H, W, C] -> [B, C, H, W]
        x = x.permute(0, 3, 1, 2)

        x = identity + self.drop_path(x)

        return x


class ConvNeXtV2(nn.Module):
    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.0
    ):
        super().__init__()

        self.depths = depths
        self.dims = dims

        self.downsample_layers = nn.ModuleList()

        # Stem: 4x4 Conv, stride 4
        stem = nn.Sequential(
            nn.Conv2d(
                in_chans,
                dims[0],
                kernel_size=4,
                stride=4
            ),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )

        self.downsample_layers.append(stem)

        # Stage 2, 3, 4 downsample
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(
                    dims[i],
                    dims[i + 1],
                    kernel_size=2,
                    stride=2
                )
            )

            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()

        dp_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        cur = 0

        for i in range(4):
            stage = nn.Sequential(
                *[
                    ConvNeXtV2Block(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j]
                    )
                    for j in range(depths[i])
                ]
            )

            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        # Global average pooling
        x = x.mean(dim=(-2, -1))

        x = self.norm(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)

            if m.bias is not None:
                nn.init.zeros_(m.bias)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_ConvNeXtV2_Tiny(
        num_classes=10,
        img_channels=3
    ).to(device)

    x = torch.randn(2, 3, 224, 224).to(device)

    y = model(x)

    print("Model: ConvNeXtV2-Tiny")
    print("Input:", x.shape)
    print("Output:", y.shape)
    print("Params:", count_parameters(model), "M")