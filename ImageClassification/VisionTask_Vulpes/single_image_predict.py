import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from _forModule.EConvNeXt import (
    build_EConvNeXt_Mini,
    build_EConvNeXt_Tiny,
    build_EConvNeXt_Small,
)

 

MODEL_VARIANT = "mini"
 
PTH_PATH = "_forModule/FoxSpecies_econvnext_mini_best.pth"
IMAGE_PATH = "_forModule/test.jpg"
CLASS_NAMES_PATH = "_forModule/classes.txt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 224
 

def load_class_names(class_file_path: str):
    if not os.path.exists(class_file_path):
        raise FileNotFoundError(f"Unable to find category name file : {class_file_path}")

    with open(class_file_path, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]

    if len(class_names) == 0:
        raise ValueError("classes.txt is empty; at least one category name is required.")

    return class_names

 

def build_model(model_variant: str, num_classes: int):
    model_variant = model_variant.lower()

    if model_variant == "mini":
        return build_EConvNeXt_Mini(num_classes=num_classes, img_channels=3)

    if model_variant == "tiny":
        return build_EConvNeXt_Tiny(num_classes=num_classes, img_channels=3)

    if model_variant == "small":
        return build_EConvNeXt_Small(num_classes=num_classes, img_channels=3)

    raise ValueError(
        f"Not supported MODEL_VARIANT: {model_variant}. "
        f"Selection : mini, tiny, small"
    )


 
def load_checkpoint(model, pth_path: str, device: str):
    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"Model weight file not found : {pth_path}")

    checkpoint = torch.load(pth_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        raise TypeError("The pth format is not supported: checkpoint is not a dict or state_dict.")

 
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_key = key[len("module."):]
        else:
            new_key = key
        cleaned_state_dict[new_key] = value

    model.load_state_dict(cleaned_state_dict, strict=True)
    return model

 
def build_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_image(image_path: str):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到圖片檔案: {image_path}")

    image = Image.open(image_path).convert("RGB")
    transform = build_transform()
    tensor = transform(image)
    tensor = tensor.unsqueeze(0)
    return tensor

 
def predict():
    class_names = load_class_names(CLASS_NAMES_PATH)
    num_classes = len(class_names)

    model = build_model(
        model_variant=MODEL_VARIANT,
        num_classes=num_classes,
    )

    model = load_checkpoint(
        model=model,
        pth_path=PTH_PATH,
        device=DEVICE,
    )

    model = model.to(DEVICE)
    model.eval()

    image_tensor = load_image(IMAGE_PATH).to(DEVICE)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1)

        pred_index = torch.argmax(probs, dim=1).item()
        pred_confidence = probs[0, pred_index].item()

    pred_class_name = class_names[pred_index]

    print("Predict Result")
    print("==============")
    print(f"Image Path      : {IMAGE_PATH}")
    print(f"Model Path      : {PTH_PATH}")
    print(f"Model Variant   : {MODEL_VARIANT}")
    print(f"Class Index     : {pred_index}")
    print(f"Class Name      : {pred_class_name}")
    print(f"Confidence      : {pred_confidence * 100:.2f}%")

    return pred_class_name


if __name__ == "__main__":
    predict()