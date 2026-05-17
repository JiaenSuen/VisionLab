
from .InceptionSeries.GoogLeNet         import build_googlenet
from .InceptionSeries.InceptionV2       import build_inception2
from .InceptionSeries.InceptionV3       import build_inception3
from .InceptionSeries.InceptionV3tiny   import build_inception3_tiny
from .InceptionSeries.InceptionV4       import build_inception4
from .InceptionSeries.InceptionV4tiny   import build_inception4_tiny

from .ShortcutSeries.HighwayNet        import build_HighwayNet
from .ShortcutSeries.ResNet            import build_ResNet18,build_ResNet34
from .ShortcutSeries.ResNetV2          import ResNetV2_18,ResNetV2_34
from .ShortcutSeries.WideResNet        import build_WideResNet18,build_WideResNet34
from .ShortcutSeries.ResNeXt           import resnext50_32x4d,resnext18_tiny

from .InceptionSeries.InceptionResNetV2     import build_inception_resnet_v2
from .InceptionSeries.InceptionResNetV2tiny import build_inception_resnet_v2_tiny
from .InceptionSeries.XceptionTiny          import xception_tiny



from .ConvNeXtV2_Tiny import build_ConvNeXtV2_Tiny
from .InceptionMamba import build_InceptionMamba_Tiny
from .RepViT import build_RepViT_M1_1
from .EConvNeXt import build_EConvNeXt_Mini


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

    "convnext2_tiny" : build_ConvNeXtV2_Tiny,
    "inception_mamba_tiny"   : build_InceptionMamba_Tiny,
    "rep_vit_m1_1" : build_RepViT_M1_1,
    "e-convnext-mini" : build_EConvNeXt_Mini,
    
}
Existing_model_names =   list(modelRouter.keys())