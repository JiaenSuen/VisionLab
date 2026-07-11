import torch
import torch.nn as nn
import torch.nn.functional as F


def build_RepViT_M1_1(num_classes, img_channels=3):
    return RepViT(
        cfgs=repvit_m1_1_cfgs(),
        num_classes=num_classes,
        img_channels=img_channels
    )


def build_RepViT_M1_5(num_classes, img_channels=3):
    return RepViT(
        cfgs=repvit_m1_5_cfgs(),
        num_classes=num_classes,
        img_channels=img_channels
    )


def build_RepViT_M2_3(num_classes, img_channels=3):
    return RepViT(
        cfgs=repvit_m2_3_cfgs(),
        num_classes=num_classes,
        img_channels=img_channels
    )


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor

    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

    if new_v < 0.9 * v:
        new_v += divisor

    return new_v


class Conv2d_BN(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bn_weight_init=1.0
    ):
        super().__init__()

        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=False
            )
        )

        self.add_module(
            "bn",
            nn.BatchNorm2d(out_channels)
        )

        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0.0)

    @torch.no_grad()
    def fuse(self):
        conv = self.conv
        bn = self.bn

        fused_conv = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            bias=True,
            device=conv.weight.device
        )

        w_bn = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        fused_conv.weight.data.copy_(conv.weight * w_bn[:, None, None, None])

        b_conv = torch.zeros(conv.out_channels, device=conv.weight.device)
        b_bn = bn.bias - bn.running_mean * w_bn
        fused_conv.bias.data.copy_(b_conv + b_bn)

        return fused_conv


class BN_Linear(nn.Sequential):
    def __init__(self, in_features, out_features, bias=True, std=0.02):
        super().__init__()

        self.add_module(
            "bn",
            nn.BatchNorm1d(in_features)
        )

        self.add_module(
            "linear",
            nn.Linear(in_features, out_features, bias=bias)
        )

        nn.init.trunc_normal_(self.linear.weight, std=std)

        if bias:
            nn.init.constant_(self.linear.bias, 0.0)

    @torch.no_grad()
    def fuse(self):
        bn = self.bn
        linear = self.linear

        w_bn = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        b_bn = bn.bias - bn.running_mean * w_bn

        fused_linear = nn.Linear(
            linear.in_features,
            linear.out_features,
            bias=True,
            device=linear.weight.device
        )

        fused_linear.weight.data.copy_(linear.weight * w_bn[None, :])

        if linear.bias is None:
            fused_linear.bias.data.copy_(linear.weight @ b_bn)
        else:
            fused_linear.bias.data.copy_(linear.weight @ b_bn + linear.bias)

        return fused_linear


class SqueezeExcite(nn.Module):
    def __init__(self, channels, se_ratio=0.25):
        super().__init__()

        hidden_channels = max(1, int(channels * se_ratio))

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.avgpool(x)
        scale = self.fc(scale)

        return x * scale


class Residual(nn.Module):
    def __init__(self, module, drop=0.0):
        super().__init__()

        self.module = module
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0.0:
            keep_prob = 1.0 - self.drop

            mask = torch.rand(
                x.size(0),
                1,
                1,
                1,
                device=x.device
            ).ge_(self.drop).div(keep_prob).detach()

            return x + self.module(x) * mask

        return x + self.module(x)


class RepVGGDW(nn.Module):
    """
    RepViT depthwise re-parameterizable block.

    Training:
        DWConv3x3-BN + DWConv1x1 + Identity -> BN

    Deploy:
        can be fused approximately into one DWConv3x3.
    """
    def __init__(self, channels):
        super().__init__()

        self.channels = channels

        self.conv3x3 = Conv2d_BN(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channels
        )

        self.conv1x1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=channels,
            bias=True
        )

        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        return self.bn(self.conv3x3(x) + self.conv1x1(x) + x)

    @torch.no_grad()
    def fuse(self):
        conv3x3 = self.conv3x3.fuse()
        conv1x1 = self.conv1x1
        bn = self.bn

        conv1x1_weight = F.pad(conv1x1.weight, [1, 1, 1, 1])

        if conv1x1.bias is None:
            conv1x1_bias = torch.zeros(
                self.channels,
                device=conv1x1.weight.device
            )
        else:
            conv1x1_bias = conv1x1.bias

        identity_weight = torch.zeros_like(conv3x3.weight)
        identity_weight[:, :, 1, 1] = 1.0

        final_weight = conv3x3.weight + conv1x1_weight + identity_weight
        final_bias = conv3x3.bias + conv1x1_bias

        w_bn = bn.weight / torch.sqrt(bn.running_var + bn.eps)

        fused_conv = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=self.channels,
            bias=True,
            device=conv3x3.weight.device
        )

        fused_conv.weight.data.copy_(final_weight * w_bn[:, None, None, None])

        fused_bias = bn.bias + (final_bias - bn.running_mean) * w_bn
        fused_conv.bias.data.copy_(fused_bias)

        return fused_conv


class RepViTBlock(nn.Module):
    """
    RepViT block.

    stride = 1:
        RepVGGDW + SE
        residual channel mixer

    stride = 2:
        DWConv downsample + SE + PWConv
        residual channel mixer
    """
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        kernel_size,
        stride,
        use_se
    ):
        super().__init__()

        assert stride in [1, 2]

        self.stride = stride
        self.identity = stride == 1 and in_channels == out_channels

        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(
                    in_channels,
                    in_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(kernel_size - 1) // 2,
                    groups=in_channels
                ),
                SqueezeExcite(in_channels, se_ratio=0.25) if use_se else nn.Identity(),
                Conv2d_BN(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
            )

            self.channel_mixer = Residual(
                nn.Sequential(
                    Conv2d_BN(
                        out_channels,
                        2 * out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0
                    ),
                    nn.GELU(),
                    Conv2d_BN(
                        2 * out_channels,
                        out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bn_weight_init=0.0
                    )
                )
            )

        else:
            if not self.identity:
                raise ValueError("stride=1 requires in_channels == out_channels.")

            self.token_mixer = nn.Sequential(
                RepVGGDW(in_channels),
                SqueezeExcite(in_channels, se_ratio=0.25) if use_se else nn.Identity()
            )

            self.channel_mixer = Residual(
                nn.Sequential(
                    Conv2d_BN(
                        in_channels,
                        hidden_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0
                    ),
                    nn.GELU(),
                    Conv2d_BN(
                        hidden_channels,
                        out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bn_weight_init=0.0
                    )
                )
            )

    def forward(self, x):
        x = self.token_mixer(x)
        x = self.channel_mixer(x)

        return x


class RepViT(nn.Module):
    """
    RepViT classifier.

    forward(x):
        classification logits

    forward_feature_maps(x):
        returns feature maps for FPN:
            c2: stride 4
            c3: stride 8
            c4: stride 16
            c5: stride 32
    """
    def __init__(
        self,
        cfgs,
        num_classes=1000,
        img_channels=3
    ):
        super().__init__()

        self.cfgs = cfgs

        input_channel = _make_divisible(cfgs[0][2], 8)

        self.stem = nn.Sequential(
            Conv2d_BN(
                img_channels,
                input_channel // 2,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.GELU(),
            Conv2d_BN(
                input_channel // 2,
                input_channel,
                kernel_size=3,
                stride=2,
                padding=1
            )
        )

        stages = [[], [], [], []]
        stage_idx = 0

        current_channel = input_channel

        for k, t, c, use_se, use_hs, stride in cfgs:
            if stride == 2:
                stage_idx += 1

            output_channel = _make_divisible(c, 8)
            hidden_channel = _make_divisible(current_channel * t, 8)

            stages[stage_idx].append(
                RepViTBlock(
                    in_channels=current_channel,
                    hidden_channels=hidden_channel,
                    out_channels=output_channel,
                    kernel_size=k,
                    stride=stride,
                    use_se=bool(use_se)
                )
            )

            current_channel = output_channel

        self.stage1 = nn.Sequential(*stages[0])
        self.stage2 = nn.Sequential(*stages[1])
        self.stage3 = nn.Sequential(*stages[2])
        self.stage4 = nn.Sequential(*stages[3])

        self.out_channels = self._get_out_channels_from_cfgs(cfgs)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = BN_Linear(current_channel, num_classes)

        self.apply(self._init_weights)

    def _get_out_channels_from_cfgs(self, cfgs):
        stage_channels = []
        current_stage = 0

        for k, t, c, use_se, use_hs, stride in cfgs:
            if stride == 2:
                current_stage += 1

            c = _make_divisible(c, 8)

            if len(stage_channels) <= current_stage:
                stage_channels.append(c)
            else:
                stage_channels[current_stage] = c

        return stage_channels

    def forward_feature_maps(self, x):
        x = self.stem(x)

        c2 = self.stage1(x)
        c3 = self.stage2(c2)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)

        return [c2, c3, c4, c5]

    def forward_features(self, x):
        c2, c3, c4, c5 = self.forward_feature_maps(x)

        x = self.avgpool(c5)
        x = x.flatten(1)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.classifier(x)

        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.BatchNorm2d):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def switch_to_deploy(self):
        """
        Fuse Conv-BN and RepVGGDW modules for faster inference.

        Use only after training:
            model.eval()
            model.switch_to_deploy()
        """
        _fuse_module_inplace(self)
        return self


def _fuse_module_inplace(module):
    for name, child in list(module.named_children()):
        if hasattr(child, "fuse") and callable(child.fuse):
            setattr(module, name, child.fuse())
        else:
            _fuse_module_inplace(child)


def repvit_m1_1_cfgs():
    # k, t, c, SE, HS, stride
    return [
        [3, 2, 64, 1, 0, 1],
        [3, 2, 64, 0, 0, 1],
        [3, 2, 64, 0, 0, 1],

        [3, 2, 128, 0, 0, 2],
        [3, 2, 128, 1, 0, 1],
        [3, 2, 128, 0, 0, 1],
        [3, 2, 128, 0, 0, 1],

        [3, 2, 256, 0, 1, 2],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 0, 1, 1],

        [3, 2, 512, 0, 1, 2],
        [3, 2, 512, 1, 1, 1],
        [3, 2, 512, 0, 1, 1],
    ]


def repvit_m1_5_cfgs():
    # k, t, c, SE, HS, stride
    return [
        [3, 2, 64, 1, 0, 1],
        [3, 2, 64, 0, 0, 1],
        [3, 2, 64, 1, 0, 1],
        [3, 2, 64, 0, 0, 1],
        [3, 2, 64, 0, 0, 1],

        [3, 2, 128, 0, 0, 2],
        [3, 2, 128, 1, 0, 1],
        [3, 2, 128, 0, 0, 1],
        [3, 2, 128, 1, 0, 1],
        [3, 2, 128, 0, 0, 1],
        [3, 2, 128, 0, 0, 1],

        [3, 2, 256, 0, 1, 2],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 0, 1, 1],

        [3, 2, 512, 0, 1, 2],
        [3, 2, 512, 1, 1, 1],
        [3, 2, 512, 0, 1, 1],
        [3, 2, 512, 1, 1, 1],
        [3, 2, 512, 0, 1, 1],
    ]


def repvit_m2_3_cfgs():
    # k, t, c, SE, HS, stride
    return [
        [3, 2, 80, 1, 0, 1],
        [3, 2, 80, 0, 0, 1],
        [3, 2, 80, 1, 0, 1],
        [3, 2, 80, 0, 0, 1],
        [3, 2, 80, 1, 0, 1],
        [3, 2, 80, 0, 0, 1],
        [3, 2, 80, 0, 0, 1],

        [3, 2, 160, 0, 0, 2],
        [3, 2, 160, 1, 0, 1],
        [3, 2, 160, 0, 0, 1],
        [3, 2, 160, 1, 0, 1],
        [3, 2, 160, 0, 0, 1],
        [3, 2, 160, 1, 0, 1],
        [3, 2, 160, 0, 0, 1],
        [3, 2, 160, 0, 0, 1],

        [3, 2, 320, 0, 1, 2],
        [3, 2, 320, 1, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 1, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 1, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 1, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 1, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 1, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 1, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 1, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 1, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 1, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 1, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 1, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 1, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 1, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 1, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 0, 1, 1],

        [3, 2, 640, 0, 1, 2],
        [3, 2, 640, 1, 1, 1],
        [3, 2, 640, 0, 1, 1],
    ]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_RepViT_M1_1(
        num_classes=101,
        img_channels=3
    ).to(device)

    x = torch.randn(2, 3, 224, 224).to(device)

    model.eval()

    with torch.no_grad():
        y = model(x)
        features = model.forward_feature_maps(x)

    print("Model: RepViT-M1.1")
    print("Input:", x.shape)
    print("Output:", y.shape)
    print("Params:", count_parameters(model), "M")

    for i, f in enumerate(features, 1):
        print(f"C{i + 1}:", f.shape)

    # Optional deploy fusion test
    model.switch_to_deploy()
    model.eval()

    with torch.no_grad():
        y2 = model(x)

    print("Deploy Output:", y2.shape)