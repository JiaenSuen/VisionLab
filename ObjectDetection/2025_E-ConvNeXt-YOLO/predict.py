"""Single-image inference for E-ConvNeXt + YOLOv10-style detector."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T

from dataset import VOC_CLASSES
from eval import decode_outputs, tensor_nms
from yolo import build_yolov10_econvnext


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run single-image inference with an E-ConvNeXt + YOLOv10-style checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", required=True, help="Path to best.pth.tar or last.pth.tar")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output", default="prediction.jpg", help="Output image path")
    parser.add_argument("--variant", choices=["mini", "tiny", "small"], default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--pre-nms-topk", type=int, default=2500)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--raw-model", action="store_true", help="Use raw model weights instead of EMA weights when EMA is available")
    return parser.parse_args()


def load_checkpoint(path: str) -> Dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a training checkpoint dictionary.")
    return checkpoint


def select_state_dict(checkpoint: Dict[str, object], prefer_ema: bool = True) -> Dict[str, torch.Tensor]:
    ema = checkpoint.get("ema")
    if prefer_ema and isinstance(ema, dict) and isinstance(ema.get("ema"), dict):
        return ema["ema"]
    model_state = checkpoint.get("model")
    if not isinstance(model_state, dict):
        raise KeyError("Checkpoint does not contain a 'model' state dictionary.")
    return model_state


def preprocess(image: Image.Image, image_size: int) -> torch.Tensor:
    transform = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transform(image).unsqueeze(0)


def class_name(class_id: int, num_classes: int) -> str:
    if num_classes == 20:
        return VOC_CLASSES.get(class_id, str(class_id))
    return str(class_id)


def draw_detections(image: Image.Image, detections: torch.Tensor, num_classes: int) -> Image.Image:
    rendered = image.copy()
    draw = ImageDraw.Draw(rendered)
    font = ImageFont.load_default()
    width, height = rendered.size

    for row in detections.detach().cpu().tolist():
        cls, score, cx, cy, bw, bh = row
        x1 = max(0.0, (cx - bw / 2.0) * width)
        y1 = max(0.0, (cy - bh / 2.0) * height)
        x2 = min(float(width), (cx + bw / 2.0) * width)
        y2 = min(float(height), (cy + bh / 2.0) * height)
        label = f"{class_name(int(cls), num_classes)} {score:.2f}"
        draw.rectangle((x1, y1, x2, y2), outline="white", width=2)
        text_box = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle(text_box, fill="black")
        draw.text((x1, y1), label, fill="white", font=font)
    return rendered


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(args.weights)
    saved_args = checkpoint.get("args", {}) if isinstance(checkpoint.get("args"), dict) else {}

    variant = args.variant or str(saved_args.get("variant", "mini"))
    num_classes = args.num_classes or int(saved_args.get("num_classes", 20))
    image_size = args.image_size or int(saved_args.get("image_size", 448))
    drop_path_rate = float(saved_args.get("drop_path_rate", 0.05))

    model = build_yolov10_econvnext(
        num_classes=num_classes,
        variant=variant,
        drop_path_rate=drop_path_rate,
    )
    model.load_state_dict(select_state_dict(checkpoint, prefer_ema=not args.raw_model), strict=True)
    model.to(args.device).eval()

    image = Image.open(args.image).convert("RGB")
    tensor = preprocess(image, image_size).to(args.device)

    with torch.inference_mode():
        outputs = model(tensor, branch="one2one")
        decoded = decode_outputs(
            outputs,
            score_threshold=args.score_threshold,
            topk=args.pre_nms_topk,
            branch="one2one",
        )[0]
        detections = tensor_nms(
            decoded.float(),
            iou_threshold=args.nms_iou,
            max_detections=args.max_detections,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    draw_detections(image, detections, num_classes).save(output_path)
    print(f"Saved {len(detections)} detections to {output_path}")


if __name__ == "__main__":
    main()
