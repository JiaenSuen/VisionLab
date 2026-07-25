import torch
import torch.nn as nn


class FireModule(nn.Module):
 
    def __init__(
        self,
        in_channels,
        squeeze_channels,
        expand_1x1_channels,
        expand_3x3_channels
    ):
        super().__init__()

        self.squeeze = nn.Conv2d(
            in_channels,
            squeeze_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.expand_1x1 = nn.Conv2d(
            squeeze_channels,
            expand_1x1_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.expand_3x3 = nn.Conv2d(
            squeeze_channels,
            expand_3x3_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.squeeze(x))

        expand_1x1 = self.relu(self.expand_1x1(x))
        expand_3x3 = self.relu(self.expand_3x3(x))

        # Concatenate the two expand branches along channel dimension.
        x = torch.cat([expand_1x1, expand_3x3], dim=1)

        return x


class SqueezeNet(nn.Module):
    def __init__(
        self,
        image_channels=3,
        num_classes=1000,
        dataset="imagenet",
        dropout=0.5
    ):
        super().__init__()

        dataset = dataset.lower()

        if dataset not in ["imagenet", "cifar"]:
            raise ValueError(
                "dataset must be either 'imagenet' or 'cifar'"
            )

        self.dataset = dataset

        # ImageNet uses the original large-input stem.
        if dataset == "imagenet":
            self.stem = nn.Sequential(
                nn.Conv2d(
                    image_channels,
                    96,
                    kernel_size=7,
                    stride=2,
                    padding=0
                ),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(
                    kernel_size=3,
                    stride=2,
                    ceil_mode=True
                )
            )

        # CIFAR images are only 32×32, so early aggressive
        # downsampling should be avoided.
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(
                    image_channels,
                    96,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.ReLU(inplace=True)
            )

        self.fire2 = FireModule(
            in_channels=96,
            squeeze_channels=16,
            expand_1x1_channels=64,
            expand_3x3_channels=64
        )

        self.fire3 = FireModule(
            in_channels=128,
            squeeze_channels=16,
            expand_1x1_channels=64,
            expand_3x3_channels=64
        )

        self.fire4 = FireModule(
            in_channels=128,
            squeeze_channels=32,
            expand_1x1_channels=128,
            expand_3x3_channels=128
        )

        self.maxpool4 = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            ceil_mode=True
        )

        self.fire5 = FireModule(
            in_channels=256,
            squeeze_channels=32,
            expand_1x1_channels=128,
            expand_3x3_channels=128
        )

        self.fire6 = FireModule(
            in_channels=256,
            squeeze_channels=48,
            expand_1x1_channels=192,
            expand_3x3_channels=192
        )

        self.fire7 = FireModule(
            in_channels=384,
            squeeze_channels=48,
            expand_1x1_channels=192,
            expand_3x3_channels=192
        )

        self.fire8 = FireModule(
            in_channels=384,
            squeeze_channels=64,
            expand_1x1_channels=256,
            expand_3x3_channels=256
        )

        self.maxpool8 = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            ceil_mode=True
        )

        self.fire9 = FireModule(
            in_channels=512,
            squeeze_channels=64,
            expand_1x1_channels=256,
            expand_3x3_channels=256
        )

        self.dropout = nn.Dropout(p=dropout)

        # The original SqueezeNet replaces a fully connected layer
        # with a 1×1 convolutional classifier.
        self.conv10 = nn.Conv2d(
            in_channels=512,
            out_channels=num_classes,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)

        x = self.fire2(x)
        x = self.fire3(x)
        x = self.fire4(x)
        x = self.maxpool4(x)

        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)
        x = self.maxpool8(x)

        x = self.fire9(x)

        x = self.dropout(x)
        x = self.relu(self.conv10(x))

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)

        return x

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if module is self.conv10:
                    nn.init.normal_(
                        module.weight,
                        mean=0.0,
                        std=0.01
                    )
                else:
                    nn.init.kaiming_uniform_(
                        module.weight,
                        mode="fan_in",
                        nonlinearity="relu"
                    )

                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


def SqueezeNetImageNet(num_classes=1000, image_channels=3):
    return SqueezeNet(
        image_channels=image_channels,
        num_classes=num_classes,
        dataset="imagenet"
    )


def SqueezeNetCIFAR10(num_classes=10,image_channels=3):
    return SqueezeNet(
        image_channels=image_channels,
        num_classes=num_classes,
        dataset="cifar"
    )


def SqueezeNetCIFAR100(num_classes=100,image_channels=3):
    return SqueezeNet(
        image_channels=image_channels,
        num_classes=num_classes,
        dataset="cifar"
    )


if __name__ == "__main__":
    imagenet_model = SqueezeNetImageNet(num_classes=1000)
    cifar10_model  = SqueezeNetCIFAR10()

    imagenet_input = torch.randn(4, 3, 224, 224)
    cifar_input = torch.randn(4, 3, 32, 32)

    imagenet_output = imagenet_model(imagenet_input)
    cifar_output = cifar10_model(cifar_input)

    print("ImageNet output:", imagenet_output.shape)
    print("CIFAR-10 output:", cifar_output.shape)