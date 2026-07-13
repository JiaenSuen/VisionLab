"""Run Grounding DINO Tiny on one local image."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from grounding_dino_api import GroundingDINOAPI, GroundingDINOConfig


# Edit these settings directly for a simple single-image experiment.
IMAGE_PATH = Path("input.jpg")
OUTPUT_PATH = Path("outputs/result.jpg")
CATEGORIES = ["fox"]

BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
INPUT_SHORT_SIDE = 640
INPUT_LONG_SIDE = 960
DEVICE = "auto"
DTYPE = "float32"


def load_image(image_path: str | Path) -> Image.Image:
    """Load one local image as an RGB PIL image."""

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path.resolve()}")

    with Image.open(path) as image:
        return image.convert("RGB")


def main() -> None:
    image = load_image(IMAGE_PATH)

    config = GroundingDINOConfig(
        device=DEVICE,
        dtype=DTYPE,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        input_short_side=INPUT_SHORT_SIDE,
        input_long_side=INPUT_LONG_SIDE,
    )

    detector = GroundingDINOAPI(config)
    result = detector.detect(
        image=image,
        categories=CATEGORIES,
        output_path=OUTPUT_PATH,
    )

    print(f"Prompt: {result.prompt}")
    print(f"Inference time: {result.elapsed_ms:.1f} ms")
    print(f"Detections: {len(result.detections)}")

    for index, detection in enumerate(result.detections, start=1):
        print(
            f"{index}: label={detection.label}, "
            f"score={detection.score:.3f}, box={detection.box}"
        )

    print(f"Annotated image saved to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
