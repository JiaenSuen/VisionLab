from .resnets import build_resnet18,build_resnet34

Existing_model_names = ["resnet18", "resnet34", "resnet50"]
def modelRouter(modelName=None,num_classes=None):
    if modelName is None or num_classes is None:
        raise ValueError("Model name and num_classes must be provided.")
    modelName = modelName.lower()
    # RestNet Router 
    if modelName == "resnet18":
        return build_resnet18(num_classes=num_classes)
    elif modelName == "resnet34":
        return build_resnet34(num_classes=num_classes)
 