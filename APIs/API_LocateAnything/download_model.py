"""Download LocateAnything model files into this project."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIRECTORY = PROJECT_ROOT / "models" / "LocateAnything-3B"
CACHE_DIRECTORY = PROJECT_ROOT / ".cache" / "huggingface"
MODEL_ID = "nvidia/LocateAnything-3B"

# Keep both model weights and Hugging Face cache data inside the project.
os.environ.setdefault("HF_HOME", str(CACHE_DIRECTORY))
os.environ.setdefault("HF_HUB_CACHE", str(CACHE_DIRECTORY / "hub"))

from huggingface_hub import snapshot_download


def main() -> None:
    """Download or resume downloading the complete model repository."""
    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {MODEL_ID}")
    print(f"Model directory: {MODEL_DIRECTORY}")

    downloaded_path = snapshot_download(
        repo_id=MODEL_ID,
        local_dir=MODEL_DIRECTORY,
    )

    print("Model download completed.")
    print(f"Local model path: {downloaded_path}")


if __name__ == "__main__":
    main()
