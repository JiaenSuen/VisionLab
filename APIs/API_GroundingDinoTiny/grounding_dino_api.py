"""Reusable Grounding DINO Tiny API for local and embedded-oriented inference."""

from __future__ import annotations

import inspect
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


ImageInput = str | Path | Image.Image | np.ndarray
Box = tuple[int, int, int, int]


@dataclass(slots=True)
class GroundingDINOConfig:
    """Centralized settings for model loading, inference, and visualization."""

    model_id: str = "IDEA-Research/grounding-dino-tiny"
    model_path: str | Path = "models/grounding-dino-tiny"
    device: str = "auto"
    dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto"
    local_files_only: bool = True

    # Accuracy and filtering settings.
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    nms_iou_threshold: float = 0.60
    class_agnostic_nms: bool = False
    min_area_ratio: float = 0.0002
    max_detections: int = 100

    # Smaller values reduce latency and memory use at the cost of small-object recall.
    input_short_side: int = 640
    input_long_side: int = 960

    # Runtime optimization settings.
    enable_tf32: bool = True
    channels_last: bool = True
    compile_model: bool = False

    # NumPy camera frames are commonly supplied by OpenCV in BGR order.
    numpy_color_order: Literal["RGB", "BGR"] = "BGR"

    # Visualization settings.
    draw_scores: bool = True
    box_style: Literal["corners", "rectangle"] = "corners"
    box_width: int = 3
    corner_ratio: float = 0.22
    connect_corner_edges: bool = True
    dash_length: int = 10
    dash_gap: int = 7
    dash_width: int = 1
    dash_lighten_ratio: float = 0.55
    font_size: int = 18

    def __post_init__(self) -> None:
        if not 0.0 <= self.box_threshold <= 1.0:
            raise ValueError("box_threshold must be between 0 and 1.")
        if not 0.0 <= self.text_threshold <= 1.0:
            raise ValueError("text_threshold must be between 0 and 1.")
        if not 0.0 <= self.nms_iou_threshold <= 1.0:
            raise ValueError("nms_iou_threshold must be between 0 and 1.")
        if not 0.0 <= self.min_area_ratio <= 1.0:
            raise ValueError("min_area_ratio must be between 0 and 1.")
        if self.input_short_side <= 0 or self.input_long_side <= 0:
            raise ValueError("Input dimensions must be positive.")
        if self.input_long_side < self.input_short_side:
            raise ValueError("input_long_side must be >= input_short_side.")
        if self.max_detections <= 0:
            raise ValueError("max_detections must be positive.")
        if self.box_width <= 0:
            raise ValueError("box_width must be positive.")
        if self.dash_length <= 0:
            raise ValueError("dash_length must be positive.")
        if self.dash_gap < 0:
            raise ValueError("dash_gap must be zero or positive.")
        if self.dash_width <= 0:
            raise ValueError("dash_width must be positive.")
        if not 0.0 <= self.dash_lighten_ratio <= 1.0:
            raise ValueError("dash_lighten_ratio must be between 0 and 1.")


@dataclass(slots=True)
class Detection:
    """One detection in original-image pixel coordinates."""

    label: str
    score: float
    box: Box

    def as_dict(self) -> dict[str, Any]:
        return {"label": self.label, "score": self.score, "box": list(self.box)}


@dataclass(slots=True)
class DetectionResult:
    """Structured output returned by GroundingDINOAPI.detect()."""

    detections: list[Detection]
    annotated_image: Image.Image
    prompt: str
    elapsed_ms: float
    image_size: tuple[int, int]
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "detections": [item.as_dict() for item in self.detections],
            "prompt": self.prompt,
            "elapsed_ms": self.elapsed_ms,
            "image_size": list(self.image_size),
            "metadata": self.metadata,
        }


class GroundingDINOAPI:
    """Load Grounding DINO Tiny once and reuse it for many images."""

    def __init__(self, config: GroundingDINOConfig | None = None) -> None:
        self.config = config or GroundingDINOConfig()
        self.device = self._resolve_device(self.config.device)
        self.dtype = self._resolve_dtype(self.config.dtype, self.device)
        self.model_path = Path(self.config.model_path)

        if self.config.local_files_only and not (self.model_path / "config.json").exists():
            raise FileNotFoundError(
                f"Local model not found at: {self.model_path.resolve()}\n"
                "Run 'python download_model.py' before starting inference."
            )

        source = str(self.model_path) if self.model_path.exists() else self.config.model_id
        print(f"Loading processor from: {source}")
        self.processor = AutoProcessor.from_pretrained(
            source,
            local_files_only=self.config.local_files_only,
        )
        self._set_processor_size()

        print(f"Loading model on {self.device} with {self.dtype}")
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            source,
            dtype=self.dtype,
            local_files_only=self.config.local_files_only,
        )
        self.model = self.model.to(self.device).eval()

        if self.config.channels_last and self.device.type == "cuda":
            self.model = self.model.to(memory_format=torch.channels_last)
        if self.config.enable_tf32 and self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if self.config.compile_model:
            if not hasattr(torch, "compile"):
                raise RuntimeError("torch.compile requires PyTorch 2.0 or newer.")
            print("Compiling model. The first inference may take significantly longer.")
            self.model = torch.compile(self.model, mode="reduce-overhead")

    def detect(
        self,
        image: ImageInput,
        categories: str | Sequence[str],
        output_path: str | Path | None = None,
        valid_roi: Box | None = None,
        *,
        box_threshold: float | None = None,
        text_threshold: float | None = None,
    ) -> DetectionResult:
        """Detect one or more text-specified categories in an image."""

        labels = self._normalize_categories(categories)
        original = self._load_image(image)
        roi = self._validate_roi(valid_roi, original.size)
        working = original.crop(roi) if roi is not None else original
        offset_x, offset_y = (roi[0], roi[1]) if roi is not None else (0, 0)

        threshold = self.config.box_threshold if box_threshold is None else box_threshold
        phrase_threshold = self.config.text_threshold if text_threshold is None else text_threshold
        self._validate_runtime_thresholds(threshold, phrase_threshold)

        start = time.perf_counter()
        inputs = self.processor(images=working, text=labels, return_tensors="pt")
        inputs = inputs.to(self.device)
        if "pixel_values" in inputs:
            memory_format = (
                torch.channels_last
                if self.config.channels_last and self.device.type == "cuda"
                else torch.contiguous_format
            )
            inputs["pixel_values"] = inputs["pixel_values"].to(
                dtype=self.dtype,
                memory_format=memory_format,
            )

        with torch.inference_mode():
            outputs = self.model(**inputs)

        processed = self._post_process(
            outputs=outputs,
            input_ids=inputs["input_ids"],
            target_size=(working.height, working.width),
            box_threshold=threshold,
            text_threshold=phrase_threshold,
        )
        detections = self._build_detections(
            processed,
            working_size=working.size,
            offset=(offset_x, offset_y),
        )
        detections = self._apply_nms(detections)
        detections = detections[: self.config.max_detections]

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        annotated = self.draw_detections(original, detections)
        if output_path is not None:
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            annotated.save(destination)

        prompt = ". ".join(labels) + "."
        return DetectionResult(
            detections=detections,
            annotated_image=annotated,
            prompt=prompt,
            elapsed_ms=elapsed_ms,
            image_size=original.size,
            metadata={
                "model": self.config.model_id,
                "device": str(self.device),
                "dtype": str(self.dtype).replace("torch.", ""),
                "box_threshold": threshold,
                "text_threshold": phrase_threshold,
                "valid_roi": list(roi) if roi is not None else None,
            },
        )

    def ground(
        self,
        image: ImageInput,
        description: str,
        output_path: str | Path | None = None,
        valid_roi: Box | None = None,
        **kwargs: Any,
    ) -> DetectionResult:
        """Ground one natural-language referring expression in an image."""

        return self.detect(
            image=image,
            categories=[description],
            output_path=output_path,
            valid_roi=valid_roi,
            **kwargs,
        )

    def warmup(self, image_size: tuple[int, int] = (640, 480)) -> None:
        """Run one disposable inference to initialize GPU kernels."""

        width, height = image_size
        dummy = Image.new("RGB", (width, height), color=(127, 127, 127))
        self.detect(dummy, ["object"], box_threshold=0.99, text_threshold=0.99)

    def draw_detections(
        self,
        image: Image.Image,
        detections: Iterable[Detection],
    ) -> Image.Image:
        """Draw detections without changing the source image."""

        canvas = image.copy().convert("RGB")
        draw = ImageDraw.Draw(canvas)
        font = self._load_font(self.config.font_size)
        palette = [
            (0, 220, 255),
            (255, 86, 110),
            (105, 235, 125),
            (255, 190, 50),
            (175, 110, 255),
            (70, 145, 255),
        ]
        colors: dict[str, tuple[int, int, int]] = {}

        for detection in detections:
            color = colors.setdefault(
                detection.label,
                palette[len(colors) % len(palette)],
            )
            if self.config.box_style == "corners":
                self._draw_corner_box(draw, detection.box, color)
            else:
                draw.rectangle(detection.box, outline=color, width=self.config.box_width)

            caption = detection.label
            if self.config.draw_scores:
                caption += f" {detection.score:.2f}"
            self._draw_label(draw, detection.box, caption, color, font)

        return canvas

    def _set_processor_size(self) -> None:
        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is None:
            return
        image_processor.size = {
            "shortest_edge": self.config.input_short_side,
            "longest_edge": self.config.input_long_side,
        }

    def _post_process(
        self,
        outputs: Any,
        input_ids: torch.Tensor,
        target_size: tuple[int, int],
        box_threshold: float,
        text_threshold: float,
    ) -> dict[str, Any]:
        method = self.processor.post_process_grounded_object_detection
        parameters = inspect.signature(method).parameters
        kwargs: dict[str, Any] = {
            "outputs": outputs,
            "input_ids": input_ids,
            "text_threshold": text_threshold,
            "target_sizes": [target_size],
        }
        # Transformers renamed box_threshold to threshold.
        if "threshold" in parameters:
            kwargs["threshold"] = box_threshold
        else:
            kwargs["box_threshold"] = box_threshold
        return method(**kwargs)[0]

    def _build_detections(
        self,
        result: dict[str, Any],
        working_size: tuple[int, int],
        offset: tuple[int, int],
    ) -> list[Detection]:
        boxes = result.get("boxes", [])
        scores = result.get("scores", [])
        labels = result.get("text_labels")
        if labels is None:
            labels = result.get("labels", [])

        width, height = working_size
        image_area = max(width * height, 1)
        offset_x, offset_y = offset
        detections: list[Detection] = []

        for box_tensor, score_tensor, label in zip(boxes, scores, labels):
            values = box_tensor.detach().float().cpu().tolist()
            x1 = max(0, min(width, round(values[0])))
            y1 = max(0, min(height, round(values[1])))
            x2 = max(0, min(width, round(values[2])))
            y2 = max(0, min(height, round(values[3])))
            if x2 <= x1 or y2 <= y1:
                continue
            if ((x2 - x1) * (y2 - y1)) / image_area < self.config.min_area_ratio:
                continue

            detections.append(
                Detection(
                    label=str(label).strip() or "object",
                    score=float(score_tensor.detach().float().cpu().item()),
                    box=(x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y),
                )
            )
        return sorted(detections, key=lambda item: item.score, reverse=True)

    def _apply_nms(self, detections: list[Detection]) -> list[Detection]:
        if not detections or self.config.nms_iou_threshold >= 1.0:
            return detections

        kept: list[Detection] = []
        for candidate in detections:
            duplicate = False
            for existing in kept:
                same_group = self.config.class_agnostic_nms or candidate.label == existing.label
                if same_group and self._box_iou(candidate.box, existing.box) > self.config.nms_iou_threshold:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(candidate)
        return kept

    @staticmethod
    def _box_iou(first: Box, second: Box) -> float:
        x1 = max(first[0], second[0])
        y1 = max(first[1], second[1])
        x2 = min(first[2], second[2])
        y2 = min(first[3], second[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
        second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
        union = first_area + second_area - intersection
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _resolve_device(value: str) -> torch.device:
        if value == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(value)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is available.")
        return device

    @staticmethod
    def _resolve_dtype(value: str, device: torch.device) -> torch.dtype:
        if value == "auto":
            # Grounding DINO creates FP32 text features in some Transformers
            # versions. Loading the complete model as FP16 then causes a
            # Float/Half mismatch inside the text-enhancer linear layers.
            return torch.float32
        mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = mapping[value]
        if dtype != torch.float32:
            raise ValueError(
                "Grounding DINO requires dtype='float32' with this API because "
                "its text features may remain FP32. CUDA acceleration uses TF32 when enabled."
            )
        return dtype

    def _load_image(self, source: ImageInput) -> Image.Image:
        if isinstance(source, Image.Image):
            return source.convert("RGB")
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path.resolve()}")
            with Image.open(path) as image:
                return image.convert("RGB")
        if isinstance(source, np.ndarray):
            array = source
            if array.ndim == 2:
                array = np.repeat(array[..., None], 3, axis=2)
            if array.ndim != 3 or array.shape[2] not in (3, 4):
                raise ValueError("NumPy image must have shape HxW, HxWx3, or HxWx4.")
            if array.dtype != np.uint8:
                array = np.clip(array, 0, 255).astype(np.uint8)
            if array.shape[2] == 4:
                array = array[:, :, :3]
            if self.config.numpy_color_order == "BGR":
                array = array[:, :, ::-1]
            return Image.fromarray(np.ascontiguousarray(array), mode="RGB")
        raise TypeError("image must be a path, PIL image, or NumPy array.")

    @staticmethod
    def _normalize_categories(categories: str | Sequence[str]) -> list[str]:
        raw = [categories] if isinstance(categories, str) else list(categories)
        labels = [item.strip().lower().rstrip(".") for item in raw if item.strip()]
        if not labels:
            raise ValueError("At least one non-empty category is required.")
        if any("." in item for item in labels):
            raise ValueError("Pass each category as a separate list item without periods.")
        return labels

    @staticmethod
    def _validate_roi(roi: Box | None, image_size: tuple[int, int]) -> Box | None:
        if roi is None:
            return None
        x1, y1, x2, y2 = map(int, roi)
        width, height = image_size
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"valid_roi must fit inside the image size {image_size}.")
        return x1, y1, x2, y2

    @staticmethod
    def _validate_runtime_thresholds(box_threshold: float, text_threshold: float) -> None:
        if not 0.0 <= box_threshold <= 1.0:
            raise ValueError("box_threshold must be between 0 and 1.")
        if not 0.0 <= text_threshold <= 1.0:
            raise ValueError("text_threshold must be between 0 and 1.")

    @staticmethod
    def _load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
        for name in ("arial.ttf", "DejaVuSans.ttf"):
            try:
                return ImageFont.truetype(name, size=size)
            except OSError:
                continue
        return ImageFont.load_default()

    def _draw_corner_box(
        self,
        draw: ImageDraw.ImageDraw,
        box: Box,
        color: tuple[int, int, int],
    ) -> None:
        x1, y1, x2, y2 = box
        length = max(6, round(min(x2 - x1, y2 - y1) * self.config.corner_ratio))
        width = self.config.box_width
        lines = [
            (x1, y1, x1 + length, y1), (x1, y1, x1, y1 + length),
            (x2, y1, x2 - length, y1), (x2, y1, x2, y1 + length),
            (x1, y2, x1 + length, y2), (x1, y2, x1, y2 - length),
            (x2, y2, x2 - length, y2), (x2, y2, x2, y2 - length),
        ]
        for line in lines:
            draw.line(line, fill=color, width=width)

        if self.config.connect_corner_edges:
            dash_color = self._lighten_color(
                color,
                self.config.dash_lighten_ratio,
            )
            dashed_edges = [
                ((x1 + length, y1), (x2 - length, y1)),
                ((x1 + length, y2), (x2 - length, y2)),
                ((x1, y1 + length), (x1, y2 - length)),
                ((x2, y1 + length), (x2, y2 - length)),
            ]
            for start, end in dashed_edges:
                self._draw_dashed_line(
                    draw=draw,
                    start=start,
                    end=end,
                    color=dash_color,
                    width=self.config.dash_width,
                    dash_length=self.config.dash_length,
                    dash_gap=self.config.dash_gap,
                )

    @staticmethod
    def _draw_dashed_line(
        draw: ImageDraw.ImageDraw,
        start: tuple[int, int],
        end: tuple[int, int],
        color: tuple[int, int, int],
        width: int,
        dash_length: int,
        dash_gap: int,
    ) -> None:
        """Draw a dashed line between two points."""

        start_x, start_y = start
        end_x, end_y = end
        delta_x = end_x - start_x
        delta_y = end_y - start_y
        distance = math.hypot(delta_x, delta_y)

        if distance <= 0:
            return

        step = dash_length + dash_gap
        position = 0.0

        while position < distance:
            dash_end = min(position + dash_length, distance)
            start_ratio = position / distance
            end_ratio = dash_end / distance

            dash_start_point = (
                round(start_x + delta_x * start_ratio),
                round(start_y + delta_y * start_ratio),
            )
            dash_end_point = (
                round(start_x + delta_x * end_ratio),
                round(start_y + delta_y * end_ratio),
            )

            draw.line(
                (dash_start_point, dash_end_point),
                fill=color,
                width=width,
            )
            position += step

    @staticmethod
    def _lighten_color(
        color: tuple[int, int, int],
        ratio: float,
    ) -> tuple[int, int, int]:
        """Blend an RGB color toward white."""

        return tuple(
            round(channel + (255 - channel) * ratio)
            for channel in color
        )

    @staticmethod
    def _draw_label(
        draw: ImageDraw.ImageDraw,
        box: Box,
        caption: str,
        color: tuple[int, int, int],
        font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    ) -> None:
        x1, y1, _, _ = box
        bounds = draw.textbbox((0, 0), caption, font=font)
        text_width = bounds[2] - bounds[0]
        text_height = bounds[3] - bounds[1]
        top = max(0, y1 - text_height - 8)
        draw.rectangle((x1, top, x1 + text_width + 10, top + text_height + 7), fill=color)
        draw.text((x1 + 5, top + 3), caption, fill=(0, 0, 0), font=font)


__all__ = [
    "Detection",
    "DetectionResult",
    "GroundingDINOAPI",
    "GroundingDINOConfig",
]
