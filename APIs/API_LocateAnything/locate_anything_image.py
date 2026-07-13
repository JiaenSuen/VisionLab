"""Run LocateAnything on one image with the reusable project API."""

from pathlib import Path

from locate_anything_api import (
    LocateAnythingAPI,
    LocateAnythingConfig,
)

config = LocateAnythingConfig(
    generation_mode="slow",
    max_new_tokens=512,
    separate_categories=True,
    enable_edge_filter=True,
    enable_nms=True,
)

PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_PATH = PROJECT_ROOT / "input.jpg"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "output_annotated.jpg"
CATEGORIES = ["fox", "animal"]


def main() -> None:
    """Load the local model and annotate one image."""
    locator = LocateAnythingAPI(config)

    result = locator.detect(
        image=IMAGE_PATH,
        categories=["fox"],
        output_path="outputs/result.jpg",
    )

    print(f"Prompt: {result.prompt}")
    print(f"Raw answer: {result.raw_answer}")
    print(f"Number of detections: {len(result.detections)}")

    for detection in result.detections:
        print(
            f"label={detection.label}, box={detection.box}"
        )

    print(f"Annotated image saved to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
