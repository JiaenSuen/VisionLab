from .ResNet       import build_resnet18pt,build_resnet34pt,build_ResNet18
from .GoogLeNet    import build_googlenet,build_googlenet_pt
from .InceptionV2  import build_inception2
from .InceptionV3  import build_inception3,build_inception3_tiny
from .HighwayNet   import build_HighwayNet
 
modelRouter = {
    "googlenet"   : build_googlenet,
    "googlenetpt" : build_googlenet_pt,

    "inceptionv2" : build_inception2,
    "inceptionv3" : build_inception3,
    "inceptionv3-tiny" : build_inception3_tiny,

    "highwaynet23" : build_HighwayNet,

    "resnet18": build_ResNet18,
    "resnet18pt": build_resnet18pt,
    "resnet34pt": build_resnet34pt,
    
}
Existing_model_names =   list(modelRouter.keys())