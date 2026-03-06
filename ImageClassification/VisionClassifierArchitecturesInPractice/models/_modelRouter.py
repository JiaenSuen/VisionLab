
from .GoogLeNet         import build_googlenet
from .InceptionV2       import build_inception2
from .InceptionV3       import build_inception3
from .InceptionV3tiny   import build_inception3_tiny
from .InceptionV4       import build_inception4
from .InceptionV4tiny   import build_inception4_tiny

from .HighwayNet        import build_HighwayNet
from .ResNet            import build_ResNet18,build_ResNet34

from .InceptionResNetV2 import build_inception_resnet_v2
from .InceptionResNetV2tiny import build_inception_resnet_v2_tiny

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
}
Existing_model_names =   list(modelRouter.keys())