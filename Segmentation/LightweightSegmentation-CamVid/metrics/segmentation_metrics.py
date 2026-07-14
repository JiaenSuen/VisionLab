from __future__ import annotations


def format_metrics(metrics: dict, class_names: tuple[str, ...]) -> str:
    lines = [
        f"pixel_accuracy: {metrics['pixel_accuracy']:.4f}",
        f"mean_accuracy:  {metrics['mean_accuracy']:.4f}",
        f"mean_iou:       {metrics['mean_iou']:.4f}",
        "class_iou:",
    ]
    for name, value in zip(class_names, metrics["class_iou"]):
        lines.append(f"  {name:<12} {value:.4f}")
    return "\n".join(lines)

