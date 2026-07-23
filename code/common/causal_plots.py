#!/usr/bin/env python3
"""Causal-validation plots for the formal pipeline.

These plots are generated after causal validation. They use When2Tool-aligned
evaluation tables and neuron jsonl files, not raw label-distribution summaries.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TASK_TYPES = ("A", "B", "C")
SUBSETS = ("single_hop", "multi_hop")
MATRICES = ("Q", "K", "V", "O")


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def as_float(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    if key == "tool_necessary_accuracy":
        value = row.get("tool_necessary_accuracy", row.get("recall_tool", default))
    elif key == "no_tool_accuracy":
        if "no_tool_accuracy" in row:
            value = row["no_tool_accuracy"]
        else:
            value = 1.0 - float(row.get("overcall", default))
    else:
        value = row.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def finite_pearson(xs: List[float], ys: List[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return 0.0
    px, py = zip(*pairs)
    mean_x = sum(px) / len(px)
    mean_y = sum(py) / len(py)
    num = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in px))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in py))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def rows_by_task_intervention(summary_rows: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    return {(str(row["task_type"]), str(row["intervention"])): row for row in summary_rows}


def tdn_rows_for_type(neuron_dir: Path, task_type: str) -> List[Dict[str, Any]]:
    return read_jsonl(neuron_dir / task_type / "TDN_neurons.jsonl")


def neuron_count_by_layer_matrix(rows: Iterable[Dict[str, Any]]) -> Dict[Tuple[int, str], int]:
    counts: Dict[Tuple[int, str], int] = {}
    for row in rows:
        key = (int(row["layer"]), str(row["matrix"]))
        counts[key] = counts.get(key, 0) + 1
    return counts


def infer_num_layers(neuron_dir: Optional[Path], rows: Iterable[Dict[str, Any]]) -> int:
    if neuron_dir is not None:
        manifest_path = neuron_dir / "manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            matrix_dims = manifest.get("matrix_dims", {})
            if matrix_dims:
                return int(next(iter(matrix_dims.values()))[0])
    max_layer = -1
    for row in rows:
        max_layer = max(max_layer, int(row["layer"]))
    return max_layer + 1 if max_layer >= 0 else 1


def save_fig(fig: Any, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def plot_stage6_metric_bars(summary_rows: List[Dict[str, Any]], output_path: Path) -> str:
    by_key = rows_by_task_intervention(summary_rows)
    metrics = [
        ("accuracy", "Acc"),
        ("tool_accuracy", "ToolAcc"),
        ("tool_necessary_accuracy", "ToolNecessaryAcc"),
        ("no_tool_accuracy", "NoToolAcc"),
    ]
    interventions = [
        ("Base", "Base"),
        ("M-Random", "M-Random"),
        ("M-TDN", "M-TDN_c"),
    ]
    colors = {"Base": "#4c78a8", "M-Random": "#f58518", "M-TDN": "#54a24b"}
    x = list(range(len(TASK_TYPES)))
    width = 0.24
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.4), sharex=True)
    for ax, (metric_key, metric_name) in zip(axes.ravel(), metrics):
        for idx, (label, intervention_key) in enumerate(interventions):
            values = []
            for task_type in TASK_TYPES:
                real_key = f"M-TDN_{task_type}" if intervention_key == "M-TDN_c" else intervention_key
                values.append(as_float(by_key.get((task_type, real_key), {}), metric_key, float("nan")))
            positions = [value + (idx - 1) * width for value in x]
            ax.bar(positions, values, width=width, label=label, color=colors[label], alpha=0.9)
        ax.set_title(metric_name)
        ax.set_ylim(0.0, 1.05)
        ax.grid(axis="y", alpha=0.25)
        ax.set_xticks(x)
        ax.set_xticklabels(TASK_TYPES)
    axes[0][0].legend(frameon=False, ncols=3, loc="upper left")
    fig.suptitle("Stage 6 single-type causal validation metrics", fontsize=14)
    return save_fig(fig, output_path)


def plot_stage6_delta(summary_rows: List[Dict[str, Any]], output_path: Path) -> str:
    by_key = rows_by_task_intervention(summary_rows)
    metrics = [
        ("accuracy", "DeltaAcc"),
        ("tool_accuracy", "DeltaToolAcc"),
        ("tool_call_rate", "DeltaTCR"),
        ("tool_necessary_accuracy", "DeltaToolNecessaryAcc"),
        ("no_tool_accuracy", "DeltaNoToolAcc"),
    ]
    intervention_specs = [
        ("M-Random", "M-Random", "#f58518"),
        ("M-TDN", "M-TDN_c", "#54a24b"),
    ]
    x = list(range(len(TASK_TYPES)))
    width = 0.32
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 7.4), sharex=True)
    for ax, (metric_key, metric_name) in zip(axes.ravel(), metrics):
        for idx, (label, intervention_key, color) in enumerate(intervention_specs):
            values = []
            for task_type in TASK_TYPES:
                base = by_key.get((task_type, "Base"), {})
                real_key = f"M-TDN_{task_type}" if intervention_key == "M-TDN_c" else intervention_key
                row = by_key.get((task_type, real_key), {})
                values.append(as_float(row, metric_key, 0.0) - as_float(base, metric_key, 0.0))
            positions = [value + (idx - 0.5) * width for value in x]
            ax.bar(positions, values, width=width, label=label, color=color, alpha=0.9)
        ax.axhline(0.0, color="#333333", linewidth=0.8)
        ax.set_title(metric_name)
        ax.grid(axis="y", alpha=0.25)
        ax.set_xticks(x)
        ax.set_xticklabels(TASK_TYPES)
    axes.ravel()[-1].axis("off")
    axes[0][0].legend(frameon=False, ncols=2, loc="upper left")
    fig.suptitle("Stage 6 intervention deltas against Base", fontsize=14)
    return save_fig(fig, output_path)


def plot_stage6_tdn_count_relation(summary_rows: List[Dict[str, Any]], neuron_dir: Path, output_path: Path) -> str:
    by_key = rows_by_task_intervention(summary_rows)
    counts = [len(tdn_rows_for_type(neuron_dir, task_type)) for task_type in TASK_TYPES]
    base_acc = [as_float(by_key.get((task_type, "Base"), {}), "accuracy") for task_type in TASK_TYPES]
    tdn_acc = [
        as_float(by_key.get((task_type, f"M-TDN_{task_type}"), {}), "accuracy")
        for task_type in TASK_TYPES
    ]
    base_tool_acc = [as_float(by_key.get((task_type, "Base"), {}), "tool_accuracy") for task_type in TASK_TYPES]
    tdn_tool_acc = [
        as_float(by_key.get((task_type, f"M-TDN_{task_type}"), {}), "tool_accuracy")
        for task_type in TASK_TYPES
    ]
    delta_acc = [tdn_acc[i] - base_acc[i] for i in range(len(TASK_TYPES))]
    delta_tool_acc = [tdn_tool_acc[i] - base_tool_acc[i] for i in range(len(TASK_TYPES))]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.9))
    x = list(range(len(TASK_TYPES)))

    axes[0].bar(x, counts, width=0.52, color="#72b7b2", label="TDN count")
    axes[0].set_ylabel("TDN count")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(TASK_TYPES)
    axes[0].grid(axis="y", alpha=0.25)
    ax0b = axes[0].twinx()
    ax0b.plot(x, base_acc, color="#4c78a8", marker="o", linewidth=2.0, label="Base Acc")
    ax0b.plot(x, tdn_acc, color="#54a24b", marker="s", linewidth=2.0, label="M-TDN Acc")
    ax0b.plot(x, base_tool_acc, color="#9ecae9", marker="o", linestyle="--", linewidth=1.8, label="Base ToolAcc")
    ax0b.plot(x, tdn_tool_acc, color="#98df8a", marker="s", linestyle="--", linewidth=1.8, label="M-TDN ToolAcc")
    ax0b.set_ylim(0.0, 1.05)
    ax0b.set_ylabel("Metric value")
    lines, labels = axes[0].get_legend_handles_labels()
    lines_b, labels_b = ax0b.get_legend_handles_labels()
    axes[0].legend(lines + lines_b, labels + labels_b, frameon=False, fontsize=8, loc="upper left")
    axes[0].set_title("TDN count and causal-validation metrics")

    axes[1].bar([value - 0.18 for value in x], delta_acc, width=0.32, color="#54a24b", label="DeltaAcc")
    axes[1].bar([value + 0.18 for value in x], delta_tool_acc, width=0.32, color="#e45756", label="DeltaToolAcc")
    axes[1].axhline(0.0, color="#333333", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(TASK_TYPES)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False)
    axes[1].set_title(
        "Causal effect vs count "
        f"(r_count_deltaAcc={finite_pearson([float(v) for v in counts], delta_acc):.2f})"
    )
    fig.suptitle("Stage 6 TDN abundance and causal effect", fontsize=14)
    return save_fig(fig, output_path)


def plot_stage6_tdn_heatmap(neuron_dir: Path, output_path: Path) -> str:
    rows_by_type = {task_type: tdn_rows_for_type(neuron_dir, task_type) for task_type in TASK_TYPES}
    all_rows = [row for rows in rows_by_type.values() for row in rows]
    num_layers = infer_num_layers(neuron_dir, all_rows)
    vmax = 1
    matrices = list(MATRICES)
    heatmaps: Dict[str, List[List[int]]] = {}
    for task_type, rows in rows_by_type.items():
        counts = neuron_count_by_layer_matrix(rows)
        matrix = [
            [counts.get((layer, matrix_name), 0) for matrix_name in matrices]
            for layer in range(num_layers)
        ]
        heatmaps[task_type] = matrix
        vmax = max(vmax, max((max(row) for row in matrix), default=0))

    fig, axes = plt.subplots(1, len(TASK_TYPES), figsize=(12.6, 9.0), sharey=True)
    if len(TASK_TYPES) == 1:
        axes = [axes]
    last_image = None
    for ax, task_type in zip(axes, TASK_TYPES):
        last_image = ax.imshow(heatmaps[task_type], aspect="auto", cmap="YlGnBu", vmin=0, vmax=vmax)
        ax.set_title(f"Type {task_type}")
        ax.set_xticks(range(len(matrices)))
        ax.set_xticklabels(matrices)
        ax.set_xlabel("Matrix")
        ax.set_yticks(range(num_layers))
        ax.set_ylabel("Layer")
    if last_image is not None:
        fig.colorbar(last_image, ax=axes, shrink=0.72, label="TDN count")
    fig.suptitle("Stage 6 single-type TDN heatmap", fontsize=14)
    return save_fig(fig, output_path)


def plot_stage6_single_type(output_root: Path, neuron_dir: Path, summary_rows: List[Dict[str, Any]]) -> List[str]:
    figures_dir = output_root / "figures"
    figures = [
        plot_stage6_metric_bars(summary_rows, figures_dir / "metric_bars.png"),
        plot_stage6_delta(summary_rows, figures_dir / "delta_from_base.png"),
        plot_stage6_tdn_count_relation(summary_rows, neuron_dir, figures_dir / "tdn_count_vs_causal_effect.png"),
        plot_stage6_tdn_heatmap(neuron_dir, figures_dir / "tdn_heatmap.png"),
    ]
    write_json(
        figures_dir / "plot_manifest.json",
        {
            "stage": "stage6_single_type_causal_validation_plots",
            "source": "summary_table.csv and stage-5 TDN_neurons.jsonl",
            "metrics": ["Acc", "ToolAcc", "ToolNecessaryAcc", "NoToolAcc", "TCR"],
            "figures": figures,
        },
    )
    return figures


def plot_stage8_metric_bars(effect_rows: List[Dict[str, Any]], output_path: Path) -> str:
    rows = [row for row in effect_rows if row.get("intervention_group") in {"Base", "M-Random", "M-CTD", "M-Private_c"}]
    interventions = [row["intervention_group"] for row in rows]
    metrics = [
        ("mean_accuracy", "MeanAcc"),
        ("mean_tool_accuracy", "MeanToolAcc"),
        ("mean_tool_necessary_accuracy", "MeanToolNecessaryAcc"),
        ("mean_no_tool_accuracy", "MeanNoToolAcc"),
    ]
    x = list(range(len(interventions)))
    width = 0.18
    colors = ["#4c78a8", "#e45756", "#54a24b", "#f58518"]
    fig, ax = plt.subplots(figsize=(11.6, 5.2))
    for idx, (metric_key, metric_name) in enumerate(metrics):
        values = [as_float(row, metric_key) for row in rows]
        positions = [value + (idx - 1.5) * width for value in x]
        ax.bar(positions, values, width=width, color=colors[idx], label=metric_name, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(interventions, rotation=12)
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=2)
    ax.set_title("Stage 8 cross-type causal-validation metrics")
    return save_fig(fig, output_path)


def plot_stage8_delta(effect_rows: List[Dict[str, Any]], output_path: Path) -> str:
    rows = [row for row in effect_rows if row.get("intervention_group") != "Base"]
    interventions = [row["intervention_group"] for row in rows]
    metrics = [
        ("delta_accuracy", "DeltaAcc"),
        ("delta_tcr", "DeltaTCR"),
        ("delta_tool_accuracy", "DeltaToolAcc"),
        ("delta_tool_necessary_accuracy", "DeltaToolNecessaryAcc"),
        ("delta_no_tool_accuracy", "DeltaNoToolAcc"),
    ]
    x = list(range(len(interventions)))
    width = 0.15
    colors = ["#4c78a8", "#f58518", "#e45756", "#54a24b", "#b279a2"]
    fig, ax = plt.subplots(figsize=(12.2, 5.2))
    for idx, (metric_key, metric_name) in enumerate(metrics):
        values = [as_float(row, metric_key) for row in rows]
        positions = [value + (idx - 2) * width for value in x]
        ax.bar(positions, values, width=width, color=colors[idx], label=metric_name, alpha=0.9)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(interventions, rotation=12)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=3, fontsize=8)
    ax.set_title("Stage 8 cross-type deltas against Base")
    return save_fig(fig, output_path)


def plot_ctd_heatmap(shared_subset_dir: Path, single_type_subset_dir: Optional[Path], output_path: Path) -> str:
    rows = read_jsonl(shared_subset_dir / "CTD_neurons.jsonl")
    num_layers = infer_num_layers(single_type_subset_dir, rows)
    counts = neuron_count_by_layer_matrix(rows)
    matrices = list(MATRICES)
    heatmap = [
        [counts.get((layer, matrix_name), 0) for matrix_name in matrices]
        for layer in range(num_layers)
    ]
    vmax = max(1, max((max(row) for row in heatmap), default=0))
    fig, ax = plt.subplots(figsize=(4.8, 8.8))
    image = ax.imshow(heatmap, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(matrices)))
    ax.set_xticklabels(matrices)
    ax.set_yticks(range(num_layers))
    ax.set_xlabel("Matrix")
    ax.set_ylabel("Layer")
    ax.set_title("CTD heatmap")
    fig.colorbar(image, ax=ax, shrink=0.75, label="CTD count")
    return save_fig(fig, output_path)


def plot_stage8_subset(
    subset_output: Path,
    shared_subset_dir: Path,
    single_type_subset_dir: Path,
    summary_rows: List[Dict[str, Any]],
    effect_rows: List[Dict[str, Any]],
) -> List[str]:
    _ = summary_rows
    figures_dir = subset_output / "figures"
    figures = [
        plot_stage8_metric_bars(effect_rows, figures_dir / "cross_type_metric_bars.png"),
        plot_stage8_delta(effect_rows, figures_dir / "cross_type_delta_from_base.png"),
        plot_ctd_heatmap(shared_subset_dir, single_type_subset_dir, figures_dir / "ctd_heatmap.png"),
    ]
    write_json(
        figures_dir / "plot_manifest.json",
        {
            "stage": "formal_stage8_cross_type_causal_validation_plots",
            "source": "summary_table.csv, cross_type_effects.csv, and stage-7 CTD_neurons.jsonl",
            "metrics": ["MeanAcc", "MeanToolAcc", "MeanToolNecessaryAcc", "MeanNoToolAcc", "DeltaAcc", "DeltaTCR"],
            "figures": figures,
        },
    )
    return figures


def plot_stage8_overview(output_root: Path, effect_rows: List[Dict[str, Any]]) -> List[str]:
    figures_dir = output_root / "figures"
    subsets = [subset for subset in SUBSETS if any(row.get("subset_scope") == subset for row in effect_rows)]
    metrics = [("delta_accuracy", "DeltaAcc"), ("delta_tcr", "DeltaTCR"), ("delta_tool_accuracy", "DeltaToolAcc")]
    fig, axes = plt.subplots(1, max(1, len(subsets)), figsize=(6.2 * max(1, len(subsets)), 5.0), sharey=True)
    if len(subsets) == 1:
        axes = [axes]
    for ax, subset in zip(axes, subsets):
        rows = [row for row in effect_rows if row.get("subset_scope") == subset and row.get("intervention_group") != "Base"]
        interventions = [row["intervention_group"] for row in rows]
        x = list(range(len(interventions)))
        width = 0.22
        for idx, (metric_key, metric_name) in enumerate(metrics):
            values = [as_float(row, metric_key) for row in rows]
            positions = [value + (idx - 1) * width for value in x]
            ax.bar(positions, values, width=width, label=metric_name, alpha=0.9)
        ax.axhline(0.0, color="#333333", linewidth=0.8)
        ax.set_title(subset)
        ax.set_xticks(x)
        ax.set_xticklabels(interventions, rotation=12)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, ncols=3, fontsize=8)
    fig.suptitle("Stage 8 cross-type causal effects by subset", fontsize=14)
    figure = save_fig(fig, figures_dir / "cross_type_effects_by_subset.png")
    write_json(
        figures_dir / "plot_manifest.json",
        {
            "stage": "formal_stage8_cross_type_causal_validation_overview_plots",
            "source": "cross_type_effects.csv",
            "metrics": ["DeltaAcc", "DeltaTCR", "DeltaToolAcc"],
            "figures": [figure],
        },
    )
    return [figure]
