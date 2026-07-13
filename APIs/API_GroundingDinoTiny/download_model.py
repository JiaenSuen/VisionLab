"""Download Grounding DINO Tiny into the project directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
DEFAULT_OUTPUT = Path("models/grounding-dino-tiny")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--revision", default="main")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.model_id} to {args.output_dir.resolve()}")
    path = snapshot_download(
        repo_id=args.model_id,
        revision=args.revision,
        local_dir=args.output_dir,
    )
    print(f"Model download completed: {Path(path).resolve()}")


if __name__ == "__main__":
    main()
