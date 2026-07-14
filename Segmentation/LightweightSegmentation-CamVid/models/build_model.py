from .fast_scnn import FastSCNN
from .lraspp import build_lraspp_mobilenet_v3_large


def build_model(name: str, num_classes: int, pretrained: bool = True):
    normalized = name.lower().replace("-", "_")
    if normalized in {"lraspp", "lraspp_mobilenet_v3", "lraspp_mobilenet_v3_large"}:
        return build_lraspp_mobilenet_v3_large(num_classes, pretrained)
    if normalized == "fast_scnn":
        return FastSCNN(num_classes)
    raise ValueError(f"Unknown model: {name}")

