"""Configurable API for local LocateAnything image grounding and detection."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, TypeAlias


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "LocateAnything-3B"
DEFAULT_CACHE_PATH = PROJECT_ROOT / ".cache" / "huggingface"

# Keep Hugging Face cache files inside the project by default.
os.environ.setdefault("HF_HOME", str(DEFAULT_CACHE_PATH))
os.environ.setdefault("HF_HUB_CACHE", str(DEFAULT_CACHE_PATH / "hub"))

try:
    import decord  # noqa: F401
    import lmdb  # noqa: F401
except ImportError as error:
    raise ImportError(
        "LocateAnything requires lmdb and decord. Install them with: "
        "python -m pip install lmdb==1.7.5 decord==0.6.0"
    ) from error

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from transformers import AutoModel, AutoProcessor, AutoTokenizer


ImageInput: TypeAlias = str | Path | Image.Image
Box: TypeAlias = tuple[int, int, int, int]
ROI: TypeAlias = tuple[int, int, int, int]


@dataclass
class LocateAnythingConfig:
    """All commonly adjusted model, generation, filtering, and drawing options."""

    # Model runtime
    model_path: Path = DEFAULT_MODEL_PATH
    device: str | None = None
    dtype: torch.dtype | None = None
    local_files_only: bool = True

    # Generation
    generation_mode: str = "slow"
    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1

    # Detection strategy
    separate_categories: bool = True

    # Basic geometry filtering
    enable_geometry_filter: bool = True
    min_width_ratio: float = 0.005
    min_height_ratio: float = 0.005
    min_area_ratio: float = 0.0001
    max_area_ratio: float = 1.0

    # Suspicious edge-box filtering
    enable_edge_filter: bool = True
    edge_margin_ratio: float = 0.005
    edge_max_area_ratio: float = 0.003
    edge_max_aspect_ratio: float = 12.0

    # Duplicate removal
    enable_nms: bool = True
    nms_iou_threshold: float = 0.75

    # Annotation style
    box_width: int = 4
    font_size: int = 20
    colors: tuple[tuple[int, int, int], ...] = field(
        default_factory=lambda: (
            (0, 220, 255),
            (255, 80, 100),
            (100, 255, 120),
            (255, 190, 40),
            (180, 100, 255),
        )
    )

    def validate(self) -> None:
        """Reject invalid settings before loading the large model."""
        if self.generation_mode not in {"fast", "hybrid", "slow"}:
            raise ValueError("generation_mode must be 'fast', 'hybrid', or 'slow'.")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive.")
        if not 0.0 <= self.nms_iou_threshold <= 1.0:
            raise ValueError("nms_iou_threshold must be between 0 and 1.")
        if not 0.0 <= self.edge_margin_ratio <= 0.5:
            raise ValueError("edge_margin_ratio must be between 0 and 0.5.")
        if not 0.0 <= self.min_area_ratio <= self.max_area_ratio <= 1.0:
            raise ValueError("Area ratios must satisfy 0 <= min <= max <= 1.")
        if not self.colors:
            raise ValueError("At least one annotation color is required.")


@dataclass(frozen=True)
class Detection:
    """One detected object represented in original-image pixel coordinates."""

    label: str
    box: Box


@dataclass
class LocateResult:
    """Structured output returned by detection and grounding methods."""

    prompts: list[str]
    raw_answers: list[str]
    detections: list[Detection]
    annotated_image: Image.Image

    @property
    def prompt(self) -> str:
        """Return all prompts as one printable string for backward compatibility."""
        return "\n".join(self.prompts)

    @property
    def raw_answer(self) -> str:
        """Return all model answers as one printable string for compatibility."""
        return "\n".join(self.raw_answers)


class LocateAnythingAPI:
    """Load one local model instance and reuse it across many images."""

    BOX_PATTERN = re.compile(
        r"<ref>(?P<label>.*?)</ref>"
        r"|"
        r"<box><(?P<x1>\d+)><(?P<y1>\d+)>"
        r"<(?P<x2>\d+)><(?P<y2>\d+)></box>"
    )

    def __init__(self, config: LocateAnythingConfig | None = None) -> None:
        self.config = config or LocateAnythingConfig()
        self.config.validate()
        self.model_path = Path(self.config.model_path).resolve()

        if not (self.model_path / "config.json").exists():
            raise FileNotFoundError(
                f"The model was not found at: {self.model_path}\n"
                "Run download_model.py before starting inference."
            )

        self.device = self.config.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = self.config.dtype or self._select_dtype(self.device)

        print(f"Loading model from: {self.model_path}")
        print(f"Device: {self.device}")
        print(f"Data type: {self.dtype}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=self.config.local_files_only,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=self.config.local_files_only,
        )
        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            local_files_only=self.config.local_files_only,
            low_cpu_mem_usage=True,
        ).to(self.device).eval()

    @staticmethod
    def _select_dtype(device: str) -> torch.dtype:
        """Select a practical inference data type for the requested device."""
        if device.startswith("cuda"):
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32

    @staticmethod
    def _open_image(image: ImageInput) -> Image.Image:
        """Read a path or PIL image, apply EXIF orientation, and convert to RGB."""
        if isinstance(image, Image.Image):
            pil_image = image.copy()
        else:
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path.resolve()}")
            pil_image = Image.open(image_path)

        return ImageOps.exif_transpose(pil_image).convert("RGB")

    @staticmethod
    def _validate_roi(roi: ROI | None, width: int, height: int) -> ROI:
        """Validate a crop ROI and return the full image when no ROI is supplied."""
        if roi is None:
            return 0, 0, width, height

        x1, y1, x2, y2 = roi
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(
                f"valid_roi must be inside a {width}x{height} image, received {roi}."
            )
        return roi

    @torch.inference_mode()
    def _predict_pil(self, image: Image.Image, prompt: str) -> str:
        """Run one model prompt against an already prepared PIL image."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        formatted_text = self.processor.py_apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[formatted_text],
            images=images,
            videos=videos,
            return_tensors="pt",
        ).to(self.device)

        generation_arguments = {
            "pixel_values": inputs["pixel_values"].to(self.dtype),
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "image_grid_hws": inputs.get("image_grid_hws"),
            "tokenizer": self.tokenizer,
            "max_new_tokens": self.config.max_new_tokens,
            "use_cache": True,
            "generation_mode": self.config.generation_mode,
            "do_sample": self.config.do_sample,
            "repetition_penalty": self.config.repetition_penalty,
            "verbose": False,
        }
        if self.config.do_sample:
            generation_arguments.update(
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )

        response = self.model.generate(**generation_arguments)
        answer = response[0] if isinstance(response, tuple) else response
        return str(answer)

    def predict(self, image: ImageInput, prompt: str) -> tuple[Image.Image, str]:
        """Run one free-form prompt without parsing or filtering its output."""
        pil_image = self._open_image(image)
        return pil_image, self._predict_pil(pil_image, prompt)

    def detect(
        self,
        image: ImageInput,
        categories: Sequence[str],
        output_path: str | Path | None = None,
        valid_roi: ROI | None = None,
    ) -> LocateResult:
        """Detect categories, filter boxes, annotate the original image, and save it."""
        clean_categories = [item.strip() for item in categories if item.strip()]
        if not clean_categories:
            raise ValueError("At least one non-empty category is required.")

        original_image = self._open_image(image)
        roi = self._validate_roi(valid_roi, original_image.width, original_image.height)
        roi_x1, roi_y1, roi_x2, roi_y2 = roi
        inference_image = original_image.crop(roi)

        if self.config.separate_categories:
            category_groups = [[category] for category in clean_categories]
        else:
            category_groups = [clean_categories]

        prompts: list[str] = []
        raw_answers: list[str] = []
        detections: list[Detection] = []

        for category_group in category_groups:
            category_text = "</c>".join(category_group)
            prompt = (
                "Locate all the instances that match the following description: "
                f"{category_text}."
            )
            answer = self._predict_pil(inference_image, prompt)
            default_label = category_group[0] if len(category_group) == 1 else "object"
            parsed = self.parse_detections(
                answer=answer,
                image_width=inference_image.width,
                image_height=inference_image.height,
                default_label=default_label,
            )
            detections.extend(self._offset_detections(parsed, roi_x1, roi_y1))
            prompts.append(prompt)
            raw_answers.append(answer)

        detections = self._apply_nms(detections)
        annotated_image = self.draw_detections(original_image, detections)

        if output_path is not None:
            save_path = Path(output_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            annotated_image.save(save_path)

        return LocateResult(prompts, raw_answers, detections, annotated_image)

    def ground(
        self,
        image: ImageInput,
        description: str,
        output_path: str | Path | None = None,
        multiple: bool = True,
        valid_roi: ROI | None = None,
    ) -> LocateResult:
        """Ground a free-form phrase and return filtered pixel-coordinate boxes."""
        description = description.strip()
        if not description:
            raise ValueError("description must not be empty.")

        original_image = self._open_image(image)
        roi = self._validate_roi(valid_roi, original_image.width, original_image.height)
        roi_x1, roi_y1, _, _ = roi
        inference_image = original_image.crop(roi)

        if multiple:
            prompt = (
                "Locate all the instances that match the following description: "
                f"{description}."
            )
        else:
            prompt = (
                "Locate a single instance that matches the following description: "
                f"{description}."
            )

        answer = self._predict_pil(inference_image, prompt)
        detections = self.parse_detections(
            answer,
            inference_image.width,
            inference_image.height,
            default_label=description,
        )
        detections = self._offset_detections(detections, roi_x1, roi_y1)
        detections = self._apply_nms(detections)
        annotated_image = self.draw_detections(original_image, detections)

        if output_path is not None:
            save_path = Path(output_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            annotated_image.save(save_path)

        return LocateResult([prompt], [answer], detections, annotated_image)

    def parse_detections(
        self,
        answer: str,
        image_width: int,
        image_height: int,
        default_label: str = "object",
    ) -> list[Detection]:
        """Parse, clamp, and geometrically filter normalized model boxes."""
        detections: list[Detection] = []
        current_label = default_label

        for match in self.BOX_PATTERN.finditer(answer):
            label = match.group("label")
            if label is not None:
                current_label = label.strip() or default_label
                continue

            normalized = (
                int(match.group("x1")),
                int(match.group("y1")),
                int(match.group("x2")),
                int(match.group("y2")),
            )
            if any(value < 0 or value > 1000 for value in normalized):
                continue

            x1 = round(normalized[0] / 1000 * image_width)
            y1 = round(normalized[1] / 1000 * image_height)
            x2 = round(normalized[2] / 1000 * image_width)
            y2 = round(normalized[3] / 1000 * image_height)
            box = (
                max(0, min(x1, image_width)),
                max(0, min(y1, image_height)),
                max(0, min(x2, image_width)),
                max(0, min(y2, image_height)),
            )

            if not self._box_is_valid(box, image_width, image_height):
                continue
            detections.append(Detection(current_label, box))

        return detections

    def _box_is_valid(self, box: Box, image_width: int, image_height: int) -> bool:
        """Apply conservative geometry and suspicious-edge filters."""
        x1, y1, x2, y2 = box
        box_width = x2 - x1
        box_height = y2 - y1
        if box_width <= 0 or box_height <= 0:
            return False

        image_area = max(image_width * image_height, 1)
        area_ratio = box_width * box_height / image_area

        if self.config.enable_geometry_filter:
            if box_width / image_width < self.config.min_width_ratio:
                return False
            if box_height / image_height < self.config.min_height_ratio:
                return False
            if not self.config.min_area_ratio <= area_ratio <= self.config.max_area_ratio:
                return False

        if self.config.enable_edge_filter:
            x_margin = image_width * self.config.edge_margin_ratio
            y_margin = image_height * self.config.edge_margin_ratio
            touches_edge = (
                x1 <= x_margin
                or y1 <= y_margin
                or x2 >= image_width - x_margin
                or y2 >= image_height - y_margin
            )
            aspect_ratio = max(
                box_width / max(box_height, 1),
                box_height / max(box_width, 1),
            )
            suspicious_edge_box = touches_edge and (
                area_ratio < self.config.edge_max_area_ratio
                or aspect_ratio > self.config.edge_max_aspect_ratio
            )
            if suspicious_edge_box:
                return False

        return True

    @staticmethod
    def _offset_detections(
        detections: Sequence[Detection],
        offset_x: int,
        offset_y: int,
    ) -> list[Detection]:
        """Map crop-relative detections back into original-image coordinates."""
        return [
            Detection(
                detection.label,
                (
                    detection.box[0] + offset_x,
                    detection.box[1] + offset_y,
                    detection.box[2] + offset_x,
                    detection.box[3] + offset_y,
                ),
            )
            for detection in detections
        ]

    @staticmethod
    def _iou(first: Box, second: Box) -> float:
        """Calculate intersection over union for two pixel-coordinate boxes."""
        intersection_x1 = max(first[0], second[0])
        intersection_y1 = max(first[1], second[1])
        intersection_x2 = min(first[2], second[2])
        intersection_y2 = min(first[3], second[3])
        intersection_width = max(0, intersection_x2 - intersection_x1)
        intersection_height = max(0, intersection_y2 - intersection_y1)
        intersection = intersection_width * intersection_height
        first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
        second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
        union = first_area + second_area - intersection
        return intersection / union if union > 0 else 0.0

    def _apply_nms(self, detections: Sequence[Detection]) -> list[Detection]:
        """Remove repeated same-label boxes without requiring confidence scores."""
        if not self.config.enable_nms:
            return list(detections)

        kept: list[Detection] = []
        for detection in detections:
            is_duplicate = any(
                detection.label.casefold() == existing.label.casefold()
                and self._iou(detection.box, existing.box)
                >= self.config.nms_iou_threshold
                for existing in kept
            )
            if not is_duplicate:
                kept.append(detection)
        return kept

    def draw_detections(
        self,
        image: Image.Image,
        detections: Sequence[Detection],
    ) -> Image.Image:
        """Draw color-coded boxes and labels on a copy of the input image."""
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        label_colors: dict[str, tuple[int, int, int]] = {}

        try:
            font = ImageFont.truetype("arial.ttf", size=self.config.font_size)
        except OSError:
            font = ImageFont.load_default()

        for detection in detections:
            label_key = detection.label.casefold()
            if label_key not in label_colors:
                color_index = len(label_colors) % len(self.config.colors)
                label_colors[label_key] = self.config.colors[color_index]

            color = label_colors[label_key]
            x1, y1, x2, y2 = detection.box
            draw.rectangle(
                (x1, y1, x2, y2),
                outline=color,
                width=self.config.box_width,
            )

            text_box = draw.textbbox((0, 0), detection.label, font=font)
            text_width = text_box[2] - text_box[0]
            text_height = text_box[3] - text_box[1]
            label_y = max(0, y1 - text_height - 10)
            draw.rectangle(
                (x1, label_y, x1 + text_width + 12, label_y + text_height + 8),
                fill=color,
            )
            draw.text(
                (x1 + 6, label_y + 3),
                detection.label,
                fill=(0, 0, 0),
                font=font,
            )

        return annotated

