from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle


# ---------------------------------------------------------------------
# Architecture specifications
#
# FLOPs are based on a single 224 × 224 input.
# Parameters are expressed in millions.
#
# Modify these values if your local implementations produce different
# parameter counts or FLOP measurements.
# ---------------------------------------------------------------------
ARCHITECTURE_SPECS = {
    "e-convnext-mini": {
        "display_name": "E-ConvNeXt-Mini",
        "flops_g": 0.93,
        "parameters_m": 7.60,
    },
    "resnet18": {
        "display_name": "ResNet-18",
        "flops_g": 1.82,
        "parameters_m": 11.69,
    },
}


def load_json(path: Path) -> dict[str, Any]:
    """Load and validate a training-history JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    required_fields = {"model", "final_report"}
    missing_fields = required_fields - data.keys()

    if missing_fields:
        raise ValueError(
            f"{path.name} is missing required fields: "
            f"{', '.join(sorted(missing_fields))}"
        )

    return data


def extract_model_metrics(data: dict[str, Any]) -> dict[str, float | str]:
    """Extract accuracy metrics and append architecture specifications."""
    model_key = str(data["model"]).lower()

    if model_key not in ARCHITECTURE_SPECS:
        supported = ", ".join(ARCHITECTURE_SPECS)
        raise ValueError(
            f"Unsupported model '{data['model']}'. "
            f"Supported model names: {supported}"
        )

    report = data["final_report"]

    train_accuracy = float(report["best_checkpoint_train_acc_no_aug"])
    test_accuracy = float(report["best_checkpoint_test_acc"])

    specs = ARCHITECTURE_SPECS[model_key]

    return {
        "model": specs["display_name"],
        "best_test_acc": test_accuracy,
        "train_acc": train_accuracy,
        "generalization_gap": train_accuracy - test_accuracy,
        "flops": float(specs["flops_g"]),
        "parameters": float(specs["parameters_m"]),
    }


def configure_font() -> None:
    """Select a commonly available publication-style font."""
    candidates = [
        "Arial",
        "Helvetica",
        "Liberation Sans",
        "DejaVu Sans",
    ]

    available_fonts = {
        font.name for font in font_manager.fontManager.ttflist
    }

    selected_font = next(
        (font for font in candidates if font in available_fonts),
        "DejaVu Sans",
    )

    plt.rcParams["font.family"] = selected_font
    plt.rcParams["axes.unicode_minus"] = False


def determine_winners(
    records: list[dict[str, float | str]],
    metrics: list[tuple[str, str, str, str, str]],
) -> dict[str, int]:
    """Return the best model index for each metric."""
    winners: dict[str, int] = {}

    for _, key, _, direction, _ in metrics:
        values = [float(record[key]) for record in records]

        best_value = max(values) if direction == "↑" else min(values)
        winners[key] = values.index(best_value)

    return winners


def create_comparison_table(
    records: list[dict[str, float | str]],
    output_path: Path,
    dpi: int = 300,
) -> None:
    """Create a compact poster-ready PNG comparison table."""
    if len(records) != 2:
        raise ValueError("Exactly two model records are required.")

    metrics = [
        (
            "Best Test Accuracy",
            "best_test_acc",
            "%",
            "↑",
            "Predictive performance",
        ),
        (
            "Training Accuracy",
            "train_acc",
            "%",
            "↑",
            "Fitting capacity",
        ),
        (
            "Generalization Gap",
            "generalization_gap",
            " pp",
            "↓",
            "Lower is better",
        ),
        (
            "FLOPs @ 224 × 224",
            "flops",
            " G",
            "↓",
            "Computational cost",
        ),
        (
            "Parameters",
            "parameters",
            " M",
            "↓",
            "Model size",
        ),
    ]

    winners = determine_winners(records, metrics)

    # Publication-style color palette
    navy = "#17324D"
    blue = "#2D6CDF"
    teal = "#149487"
    light_background = "#F4F7FA"
    alternate_row = "#EEF3F7"
    grid_color = "#C8D3DE"
    text_color = "#1D2A35"
    muted_text = "#63717E"
    winner_background = "#DDF3EA"
    winner_text = "#087866"
    white = "#FFFFFF"
    note_background = "#FFF4DD"
    note_text = "#745000"

    # Wide and compact figure for poster layouts
    figure = plt.figure(figsize=(15.5, 5.0), dpi=220)
    axis = figure.add_axes([0, 0, 1, 1])

    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    # Outer card
    axis.add_patch(
        FancyBboxPatch(
            (0.025, 0.045),
            0.95,
            0.91,
            boxstyle="round,pad=0.004,rounding_size=0.01",
            linewidth=1.0,
            edgecolor=grid_color,
            facecolor=white,
        )
    )

    # Title area
    axis.add_patch(
        FancyBboxPatch(
            (0.025, 0.825),
            0.95,
            0.13,
            boxstyle="round,pad=0.004,rounding_size=0.01",
            linewidth=0,
            facecolor=navy,
        )
    )

    axis.text(
        0.055,
        0.902,
        "FoxSpecies Model Comparison",
        fontsize=22.5,
        fontweight="bold",
        color=white,
        va="center",
    )

    axis.text(
        0.055,
        0.855,
        "Accuracy, generalization, and computational efficiency",
        fontsize=11.3,
        color="#D9E4EE",
        va="center",
    )

    axis.text(
        0.945,
        0.902,
        "TABLE 1",
        fontsize=9.8,
        fontweight="bold",
        color="#BCD0E1",
        ha="right",
        va="center",
    )

    # Table layout
    left = 0.055
    right = 0.945
    top = 0.795
    bottom = 0.165

    column_edges = [
        left,
        0.46,
        0.705,
        right,
    ]

    header_height = 0.072
    row_height = (
        top - bottom - header_height
    ) / len(metrics)

    # Header row
    for column_index in range(3):
        axis.add_patch(
            Rectangle(
                (
                    column_edges[column_index],
                    top - header_height,
                ),
                column_edges[column_index + 1]
                - column_edges[column_index],
                header_height,
                facecolor=light_background,
                edgecolor=grid_color,
                linewidth=0.9,
            )
        )

    headers = [
        "Metric",
        str(records[0]["model"]),
        str(records[1]["model"]),
    ]

    header_colors = [
        navy,
        blue,
        teal,
    ]

    for index, header in enumerate(headers):
        axis.text(
            (
                column_edges[index]
                + column_edges[index + 1]
            )
            / 2,
            top - header_height / 2,
            header,
            fontsize=12.4,
            fontweight="bold",
            color=header_colors[index],
            ha="center",
            va="center",
        )

    # Data rows
    for row_index, (
        label,
        key,
        unit,
        direction,
        description,
    ) in enumerate(metrics):
        row_top = (
            top
            - header_height
            - row_index * row_height
        )
        row_bottom = row_top - row_height

        base_color = (
            white
            if row_index % 2 == 0
            else alternate_row
        )

        # Metric label cell
        axis.add_patch(
            Rectangle(
                (column_edges[0], row_bottom),
                column_edges[1] - column_edges[0],
                row_height,
                facecolor=base_color,
                edgecolor=grid_color,
                linewidth=0.8,
            )
        )

        axis.text(
            column_edges[0] + 0.016,
            row_bottom + row_height * 0.62,
            f"{label}  {direction}",
            fontsize=10.9,
            fontweight="bold",
            color=text_color,
            va="center",
        )

        axis.text(
            column_edges[0] + 0.016,
            row_bottom + row_height * 0.28,
            description,
            fontsize=8.6,
            color=muted_text,
            va="center",
        )

        # Model value cells
        for model_index, record in enumerate(records):
            is_winner = winners[key] == model_index

            x_start = column_edges[model_index + 1]
            x_end = column_edges[model_index + 2]

            axis.add_patch(
                Rectangle(
                    (x_start, row_bottom),
                    x_end - x_start,
                    row_height,
                    facecolor=(
                        winner_background
                        if is_winner
                        else base_color
                    ),
                    edgecolor=grid_color,
                    linewidth=0.8,
                )
            )

            value = float(record[key])
            value_text = f"{value:.2f}{unit}"

            axis.text(
                (x_start + x_end) / 2,
                row_bottom + row_height * 0.58,
                value_text,
                fontsize=15.2,
                fontweight="bold",
                color=(
                    winner_text
                    if is_winner
                    else text_color
                ),
                ha="center",
                va="center",
            )

            if is_winner:
                axis.text(
                    (x_start + x_end) / 2,
                    row_bottom + row_height * 0.24,
                    "BEST",
                    fontsize=7.5,
                    fontweight="bold",
                    color=winner_text,
                    ha="center",
                    va="center",
                )

    # Automatic summary
    accuracy_difference = (
        float(records[0]["best_test_acc"])
        - float(records[1]["best_test_acc"])
    )

    flops_reduction = (
        1
        - float(records[0]["flops"])
        / float(records[1]["flops"])
    ) * 100

    parameter_reduction = (
        1
        - float(records[0]["parameters"])
        / float(records[1]["parameters"])
    ) * 100

    summary = (
        f"{records[0]['model']} achieves "
        f"{accuracy_difference:+.2f} pp test accuracy, "
        f"{flops_reduction:.0f}% fewer FLOPs, and "
        f"{parameter_reduction:.0f}% fewer parameters."
    )

    summary_y = 0.09

    axis.add_patch(
        FancyBboxPatch(
            (0.055, summary_y),
            0.89,
            0.044,
            boxstyle="round,pad=0.003,rounding_size=0.007",
            linewidth=0,
            facecolor=note_background,
        )
    )

    axis.text(
        0.075,
        summary_y + 0.022,
        summary,
        fontsize=9.5,
        color=note_text,
        va="center",
    )

    # Footnote
    axis.text(
        0.055,
        0.061,
        "Note: FLOPs are reported for one 224 × 224 input. "
        "Values may vary across counting conventions and implementations.",
        fontsize=7.9,
        color=muted_text,
        va="center",
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.05,
    )

    plt.close(figure)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a compact PNG table comparing "
            "E-ConvNeXt-Mini and ResNet-18."
        )
    )

    parser.add_argument(
        "--econvnext",
        type=Path,
        default=Path(
            "FoxSpecies_e-convnext-mini_history.json"
        ),
        help="Path to the E-ConvNeXt-Mini JSON file.",
    )

    parser.add_argument(
        "--resnet",
        type=Path,
        default=Path(
            "FoxSpecies_resnet18_history.json"
        ),
        help="Path to the ResNet-18 JSON file.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "FoxSpecies_model_comparison.png"
        ),
        help="Output PNG path.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output resolution in DPI.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    configure_font()

    econvnext_data = load_json(args.econvnext)
    resnet_data = load_json(args.resnet)

    records = [
        extract_model_metrics(econvnext_data),
        extract_model_metrics(resnet_data),
    ]

    create_comparison_table(
        records=records,
        output_path=args.output,
        dpi=args.dpi,
    )

    print(f"PNG generated successfully: {args.output.resolve()}")


if __name__ == "__main__":
    main()