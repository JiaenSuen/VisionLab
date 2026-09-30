"""Minimal forward-pass check for the YOLOv5-Nano reimplementation."""

import torch
from model import YOLOv5Nano


def main():
    model = YOLOv5Nano(in_channels=3, num_classes=20).eval()
    x = torch.randn(1, 3, 320, 320)
    with torch.no_grad():
        outputs = model(x)

    expected = [(1, 3, 10, 10, 25), (1, 3, 20, 20, 25), (1, 3, 40, 40, 25)]
    actual = [tuple(t.shape) for t in outputs]
    assert actual == expected, f"Unexpected output shapes: {actual}"

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    print("Output shapes:", actual)


if __name__ == "__main__":
    main()
