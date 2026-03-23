
from .GoogLeNet         import build_googlenet
from .InceptionV2       import build_inception2
from .InceptionV3       import build_inception3
from .InceptionV3tiny   import build_inception3_tiny
from .InceptionV4       import build_inception4
from .InceptionV4tiny   import build_inception4_tiny

from .HighwayNet        import build_HighwayNet
from .ResNet            import build_ResNet18,build_ResNet34
from .ResNetV2          import ResNetV2_18,ResNetV2_34
from .WideResNet        import build_WideResNet18,build_WideResNet34
from .ResNeXt           import resnext50_32x4d,resnext18_tiny

from .InceptionResNetV2     import build_inception_resnet_v2
from .InceptionResNetV2tiny import build_inception_resnet_v2_tiny
from .XceptionTiny          import xception_tiny

from .zTorchAPI import (
    build_googlenet_pt,
    build_resnet18pt,
    build_resnet34pt,
)
 
modelRouter = {
    "googlenet"   : build_googlenet,
    "googlenetpt" : build_googlenet_pt,

    "inceptionv2" : build_inception2,
    "inceptionv3" : build_inception3,
    "inceptionv3-tiny" : build_inception3_tiny,
    "inceptionv4" : build_inception4,
    "inceptionv4-tiny" : build_inception4_tiny,

    "highwaynet23" : build_HighwayNet,

    "resnet18": build_ResNet18,
    "resnet34": build_ResNet34,
    "resnet18pt": build_resnet18pt,
    "resnet34pt": build_resnet34pt,

    "inception-resnet-v2" : build_inception_resnet_v2,
    "inception-resnet-v2-tiny" : build_inception_resnet_v2_tiny,

    "xception-tiny" : xception_tiny,

    "wide-resnet18": build_WideResNet18,
    "wide-resnet34": build_WideResNet34,

    "resnext-18" : resnext18_tiny,
    "resnext-50" : resnext50_32x4d,

    "resnet18v2" : ResNetV2_18,
    "resnet34v2" : ResNetV2_34,

}
Existing_model_names =   list(modelRouter.keys())