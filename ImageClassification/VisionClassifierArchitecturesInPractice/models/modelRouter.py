from .resnets import build_resnet18pt,build_resnet34pt

 
modelRouter = {
    "resnet18pt": build_resnet18pt,
    "resnet34pt": build_resnet34pt,
}
Existing_model_names =   list(modelRouter.keys())