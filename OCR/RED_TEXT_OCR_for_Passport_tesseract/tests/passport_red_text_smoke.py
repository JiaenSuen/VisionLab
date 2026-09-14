"""Smoke test for the current passport/document-card red-text task."""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
result = subprocess.run(
    [sys.executable, "main.py", "examples/passport_card_sample.bmp", "--output", "outputs/test_smoke"],
    cwd=ROOT,
    capture_output=True,
    text=True,
    check=True,
)
assert "47320420" in result.stdout, result.stdout
assert (ROOT / "outputs/test_smoke/ocr_result.png").exists()
print("PASS: passport/document-card red-text pipeline produced the expected major red sequence.")
