# pip install selectivesearch
from pathlib import Path
import csv

import cv2
import selectivesearch


INPUT_PATH = Path("test.jpg")
OUTPUT_DIR = Path("output")
REGIONS_DIR = OUTPUT_DIR / "regions"

# Set to None to save every proposal.
MAX_PROPOSALS = None

# Ignore very small regions.
MIN_REGION_AREA = 200

# Number of boxes drawn on the preview image.
PREVIEW_COUNT = 200


def generate_proposals(image):
    """Generate region proposals with Selective Search."""
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    _, regions = selectivesearch.selective_search(
        rgb_image,
        scale=500,
        sigma=0.9,
        min_size=10,
    )

    proposals = []
    seen = set()

    for region in regions:
        x, y, width, height = region["rect"]

        if region["size"] < MIN_REGION_AREA:
            continue

        if width <= 0 or height <= 0:
            continue

        rect = (x, y, width, height)

        # Remove duplicate rectangles.
        if rect in seen:
            continue

        seen.add(rect)
        proposals.append(rect)

    return proposals


def main() -> None:
    """Generate and save region proposals."""
    image = cv2.imread(str(INPUT_PATH))

    if image is None:
        raise FileNotFoundError(f"Cannot read image: {INPUT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REGIONS_DIR.mkdir(parents=True, exist_ok=True)

    proposals = generate_proposals(image)

    if MAX_PROPOSALS is not None:
        proposals = proposals[:MAX_PROPOSALS]

    image_height, image_width = image.shape[:2]
    csv_path = OUTPUT_DIR / "proposals.csv"

    saved_count = 0

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["id", "x", "y", "width", "height", "filename"])

        for index, (x, y, width, height) in enumerate(proposals):
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(image_width, x1 + int(width))
            y2 = min(image_height, y1 + int(height))

            if x2 <= x1 or y2 <= y1:
                continue

            region_image = image[y1:y2, x1:x2]
            filename = f"region_{index:06d}.jpg"
            output_path = REGIONS_DIR / filename

            if not cv2.imwrite(str(output_path), region_image):
                print(f"Failed to save: {output_path}")
                continue

            writer.writerow(
                [
                    index,
                    x1,
                    y1,
                    x2 - x1,
                    y2 - y1,
                    filename,
                ]
            )

            saved_count += 1

    preview = image.copy()

    for x, y, width, height in proposals[:PREVIEW_COUNT]:
        cv2.rectangle(
            preview,
            (int(x), int(y)),
            (int(x + width), int(y + height)),
            (0, 255, 0),
            1,
        )

    preview_path = OUTPUT_DIR / "preview.jpg"

    if not cv2.imwrite(str(preview_path), preview):
        raise RuntimeError(f"Cannot save preview: {preview_path}")

    print(f"Generated proposals: {len(proposals)}")
    print(f"Saved regions: {saved_count}")
    print(f"Region directory: {REGIONS_DIR}")
    print(f"Proposal metadata: {csv_path}")
    print(f"Preview image: {preview_path}")


if __name__ == "__main__":
    main()