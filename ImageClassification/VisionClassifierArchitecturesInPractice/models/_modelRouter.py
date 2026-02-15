from .AlexNet    import build_alexnet_pt
from .ResNetsV1 import build_resnet18pt,build_resnet34pt
from .GoogLeNet import build_googlenet,build_googlenet_pt
 
modelRouter = {
    "alexnet"   : build_alexnet_pt,
    "resnet18pt": build_resnet18pt,
    "resnet34pt": build_resnet34pt,
    "googlenet": build_googlenet,
    "googlenetpt": build_googlenet_pt
}
Existing_model_names =   list(modelRouter.keys())