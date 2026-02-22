from .ResNetV1    import build_resnet18pt,build_resnet34pt
from .GoogLeNet    import build_googlenet,build_googlenet_pt
from .InceptionV2  import build_inception2
from .HighwayNet   import build_HighwayNet
 
modelRouter = {
    "googlenet"   : build_googlenet,
    "googlenetpt" : build_googlenet_pt,

    "inceptionv2" : build_inception2,

    "highwaynet23" : build_HighwayNet,

    "resnet18pt": build_resnet18pt,
    "resnet34pt": build_resnet34pt,
    
}
Existing_model_names =   list(modelRouter.keys())