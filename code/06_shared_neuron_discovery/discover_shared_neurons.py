#!/usr/bin/env python3
"""Stage 7 cross-type shared neuron discovery.

It consumes stage-5 single-type TDNs and follows the
Who Transfers Safety shared-neuron construction:

    CTD_{m,l} = TDN_{m,A,l} intersect TDN_{m,B,l} intersect TDN_{m,C,l}
    CTD_m     = union_l CTD_{m,l}

Neuron identity is the exact tuple (layer, matrix, index):
- Q/K/V neurons are projection rows.
- O neurons are output-projection columns.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.io_utils import write_json, write_jsonl  # noqa: E402

TASK_TYPES = ("A", "B", "C")
SUBSETS = ("single_hop", "multi_hop")
MATRICES = ("Q", "K", "V", "O")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-alias", default="qwen3-4b-instruct")
    parser.add_argument("--data-root", default="../tool_decision_neurons_data")
    parser.add_argument("--single-type-dir", default="", help="Default: <data-root>/neurons/<model>/single_type_by_subset")
    parser.add_argument("--causal-dir", default="", help="Optional stage-6 single_type_by_subset dir for Base metrics.")
    parser.add_argument("--output-dir", default="", help="Default: <data-root>/neurons")
    parser.add_argument("--visualization-dir", default="", help="Default: <data-root>/visualizations/<model>/shared_by_subset when --make-figures is set.")
    parser.add_argument("--make-figures", action="store_true")
    parser.add_argument("--subsets", nargs="+", default=list(SUBSETS), choices=SUBSETS)
    parser.add_argument("--skip-existing", action="store_true", help="Return early when the final manifest already exists.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()

def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def ensure_empty_or_create(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite and any(path.iterdir()):
        raise FileExistsError(f"Output directory exists and is not empty: {path}. Use --overwrite.")
    path.mkdir(parents=True, exist_ok=True)


def neuron_key(row: Dict[str, Any]) -> Tuple[int, str, int]:
    return int(row["layer"]), str(row["matrix"]), int(row["index"])


def neuron_id(layer: int, matrix: str, index: int) -> str:
    return f"L{layer:02d}.{matrix}.{index:05d}"


def row_from_key(key: Tuple[int, str, int], type_maps: Dict[str, Dict[Tuple[int, str, int], Dict[str, Any]]]) -> Dict[str, Any]:
    layer, matrix, index = key
    row: Dict[str, Any] = {
        "layer": int(layer),
        "matrix": matrix,
        "index": int(index),
        "neuron_id": neuron_id(layer, matrix, index),
    }
    scores = []
    for task_type in TASK_TYPES:
        type_row = type_maps.get(task_type, {}).get(key)
        if type_row is not None and "score_label_1" in type_row:
            score = float(type_row["score_label_1"])
            row[f"score_label_1_{task_type}"] = score
            scores.append(score)
    if scores:
        row["mean_score_label_1"] = float(sum(scores) / len(scores))
        row["min_score_label_1"] = float(min(scores))
    return row


def load_type_maps(subset_dir: Path) -> Dict[str, Dict[Tuple[int, str, int], Dict[str, Any]]]:
    type_maps: Dict[str, Dict[Tuple[int, str, int], Dict[str, Any]]] = {}
    for task_type in TASK_TYPES:
        path = subset_dir / task_type / "TDN_neurons.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing TDN file: {path}")
        rows = read_jsonl(path)
        type_maps[task_type] = {neuron_key(row): row for row in rows}
    return type_maps


def read_base_metrics(causal_dir: Path) -> Dict[Tuple[str, str], Dict[str, float]]:
    path = causal_dir / "summary_table.csv"
    if not path.exists():
        return {}
    metrics: Dict[Tuple[str, str], Dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("intervention") != "Base":
                continue
            subset = row.get("subset_scope", "combined")
            task_type = row.get("task_type", "")
            metrics[(subset, task_type)] = {
                "accuracy": float(row.get("accuracy", 0.0)),
                "tool_accuracy": float(row.get("tool_accuracy", 0.0)),
                "tool_call_rate": float(row.get("tool_call_rate", 0.0)),
                "avg_tool_calls": float(row.get("avg_tool_calls", 0.0)),
            }
    return metrics


def discover_subset(
    subset: str,
    single_type_root: Path,
    output_root: Path,
    base_metrics: Dict[Tuple[str, str], Dict[str, float]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    subset_input = single_type_root / subset
    subset_output = output_root / subset
    subset_output.mkdir(parents=True, exist_ok=True)
    type_maps = load_type_maps(subset_input)
    type_sets = {task_type: set(type_maps[task_type]) for task_type in TASK_TYPES}

    exact_shared_keys = sorted(set.intersection(*(type_sets[task_type] for task_type in TASK_TYPES)))
    shared_keys = list(exact_shared_keys)
    pairwise = {
        "AB": sorted(type_sets["A"] & type_sets["B"]),
        "AC": sorted(type_sets["A"] & type_sets["C"]),
        "BC": sorted(type_sets["B"] & type_sets["C"]),
    }
    private = {
        task_type: sorted(type_sets[task_type] - set(shared_keys))
        for task_type in TASK_TYPES
    }

    shared_rows = [row_from_key(key, type_maps) for key in shared_keys]
    write_jsonl(subset_output / "CTD_neurons.jsonl", shared_rows)
    for pair_name, keys in pairwise.items():
        write_jsonl(subset_output / f"pairwise_{pair_name}_neurons.jsonl", [row_from_key(key, type_maps) for key in keys])
    for task_type, keys in private.items():
        write_jsonl(subset_output / f"private_{task_type}_neurons.jsonl", [row_from_key(key, type_maps) for key in keys])

    layer_counts_counter = Counter(int(layer) for layer, _matrix, _index in shared_keys)
    matrix_counts_counter = Counter(str(matrix) for _layer, matrix, _index in shared_keys)
    candidate_manifest = read_json(subset_input / "manifest.json")
    num_layers = int(candidate_manifest.get("candidate_universe", {}).get("num_layers", 0))
    if num_layers <= 0:
        matrix_dims = candidate_manifest.get("matrix_dims", {})
        if matrix_dims:
            num_layers = int(next(iter(matrix_dims.values()))[0])

    layer_rows = [
        {"subset": subset, "layer": layer, "shared_count": layer_counts_counter.get(layer, 0)}
        for layer in range(num_layers)
    ]
    matrix_rows = [
        {"subset": subset, "matrix": matrix, "shared_count": matrix_counts_counter.get(matrix, 0)}
        for matrix in MATRICES
    ]
    share_rows: List[Dict[str, Any]] = []
    for task_type in TASK_TYPES:
        tdn_size = len(type_sets[task_type])
        metric = base_metrics.get((subset, task_type), {})
        share_rows.append(
            {
                "subset": subset,
                "task_type": task_type,
                "TDN_size": tdn_size,
                "CTD_size": len(shared_keys),
                "exact_CTD_size": len(exact_shared_keys),
                "share_rate": len(shared_keys) / tdn_size if tdn_size else 0.0,
                "base_accuracy": metric.get("accuracy", math.nan),
                "base_tool_accuracy": metric.get("tool_accuracy", math.nan),
                "base_tool_call_rate": metric.get("tool_call_rate", math.nan),
            }
        )
    write_csv(subset_output / "layer_counts.csv", layer_rows, ["subset", "layer", "shared_count"])
    write_csv(subset_output / "matrix_counts.csv", matrix_rows, ["subset", "matrix", "shared_count"])
    write_csv(
        subset_output / "share_rates.csv",
        share_rows,
        [
            "subset",
            "task_type",
            "TDN_size",
            "CTD_size",
            "exact_CTD_size",
            "share_rate",
            "base_accuracy",
            "base_tool_accuracy",
            "base_tool_call_rate",
        ],
    )

    subset_summary = {
        "subset": subset,
        "input_dir": str(subset_input),
        "output_dir": str(subset_output),
        "TDN_sizes": {task_type: len(type_sets[task_type]) for task_type in TASK_TYPES},
        "CTD_size": len(shared_keys),
        "exact_CTD_size": len(exact_shared_keys),
        "pairwise_sizes": {name: len(keys) for name, keys in pairwise.items()},
        "private_sizes": {task_type: len(keys) for task_type, keys in private.items()},
        "share_rates": {row["task_type"]: row["share_rate"] for row in share_rows},
        "num_layers": num_layers,
        "stage5_candidate_universe": candidate_manifest.get("candidate_universe", {}),
        "neuron_definition": "Exact (layer,matrix,index) identity; Q/K/V are projection rows and O is an output-projection column.",
        "shared_definition": "CTD_{m,l}=TDN_{m,A,l} intersect TDN_{m,B,l} intersect TDN_{m,C,l}; CTD_m=union_l CTD_{m,l}.",
    }
    write_json(subset_output / "manifest.json", subset_summary)
    return subset_summary, share_rows, layer_rows, matrix_rows


def plot_fig4_abundance(summary_rows: List[Dict[str, Any]], share_rows: List[Dict[str, Any]], output_path: Path) -> None:
    subsets = [row["subset"] for row in summary_rows]
    ctd_counts = [int(row["CTD_size"]) for row in summary_rows]
    mean_tool_acc = []
    mean_acc = []
    for subset in subsets:
        rows = [row for row in share_rows if row["subset"] == subset]
        tool_values = [float(row["base_tool_accuracy"]) for row in rows if not math.isnan(float(row["base_tool_accuracy"]))]
        acc_values = [float(row["base_accuracy"]) for row in rows if not math.isnan(float(row["base_accuracy"]))]
        mean_tool_acc.append(sum(tool_values) / len(tool_values) if tool_values else math.nan)
        mean_acc.append(sum(acc_values) / len(acc_values) if acc_values else math.nan)

    fig, ax1 = plt.subplots(figsize=(8.2, 5.2))
    x = np.arange(len(subsets))
    bars = ax1.bar(x, ctd_counts, color="#4c78a8", width=0.52, label="CTD count")
    ax1.set_xticks(x, [name.replace("_", " ") for name in subsets])
    ax1.set_ylabel("Cross-type shared neurons")
    ax1.set_title("Figure 4-style CTD Abundance vs Tool-Decision Capability")
    ax1.grid(axis="y", alpha=0.25)
    y_pad = max(max(ctd_counts, default=0) * 0.08, 0.15)
    ax1.set_ylim(0, max(max(ctd_counts, default=0) + y_pad, 1))
    for bar, value in zip(bars, ctd_counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, value + y_pad * 0.2, str(value), ha="center", fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(x, mean_tool_acc, color="#d62728", marker="o", linewidth=2.3, label="Mean Base ToolAcc")
    ax2.plot(x, mean_acc, color="#2ca02c", marker="s", linestyle="--", linewidth=2.0, label="Mean Base Acc")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Capability metric")
    handles_1, labels_1 = ax1.get_legend_handles_labels()
    handles_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(handles_1 + handles_2, labels_1 + labels_2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_share_rate_by_type(subset: str, share_rows: List[Dict[str, Any]], output_path: Path) -> None:
    rows = [row for row in share_rows if row["subset"] == subset]
    x = np.arange(len(TASK_TYPES))
    share = [float(next(row for row in rows if row["task_type"] == task_type)["share_rate"]) for task_type in TASK_TYPES]
    tool_acc = [float(next(row for row in rows if row["task_type"] == task_type)["base_tool_accuracy"]) for task_type in TASK_TYPES]
    acc = [float(next(row for row in rows if row["task_type"] == task_type)["base_accuracy"]) for task_type in TASK_TYPES]

    fig, ax1 = plt.subplots(figsize=(8.4, 5.0))
    bars = ax1.bar(x, share, color="#9467bd", width=0.52, label="ShareRate = |CTD| / |TDN_c|")
    ax1.set_xticks(x, TASK_TYPES)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Share rate")
    ax1.set_title(f"{subset}: Shared-Neuron Share vs Type Capability")
    ax1.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, share):
        ax1.text(bar.get_x() + bar.get_width() / 2, min(value + 0.035, 1.0), f"{value:.2f}", ha="center", fontsize=8)

    ax2 = ax1.twinx()
    ax2.plot(x, tool_acc, color="#d62728", marker="o", linewidth=2.3, label="Base ToolAcc")
    ax2.plot(x, acc, color="#2ca02c", marker="s", linestyle="--", linewidth=2.0, label="Base Acc")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Capability metric")
    handles_1, labels_1 = ax1.get_legend_handles_labels()
    handles_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(handles_1 + handles_2, labels_1 + labels_2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_shared_heatmap(summary_rows: List[Dict[str, Any]], output_root: Path, output_path: Path) -> None:
    num_layers = max(int(row["num_layers"]) for row in summary_rows) if summary_rows else 0
    num_layers = max(num_layers, 1)
    fig, axes = plt.subplots(1, len(summary_rows), figsize=(6.2 * max(1, len(summary_rows)), 8.8), sharey=True)
    if len(summary_rows) == 1:
        axes = [axes]
    max_count = 1
    matrices_by_subset: Dict[str, np.ndarray] = {}
    for row in summary_rows:
        subset = row["subset"]
        data = np.zeros((num_layers, len(MATRICES)), dtype=float)
        for shared in read_jsonl(output_root / subset / "CTD_neurons.jsonl"):
            data[int(shared["layer"]), MATRICES.index(str(shared["matrix"]))] += 1
        matrices_by_subset[subset] = data
        max_count = max(max_count, int(data.max()))

    fig.suptitle("Cross-Type Shared Neuron Heatmap", fontsize=15, fontweight="bold")
    for ax, row in zip(axes, summary_rows):
        subset = row["subset"]
        data = matrices_by_subset[subset]
        image = ax.imshow(data, aspect="auto", interpolation="nearest", cmap="magma", vmin=0, vmax=max_count)
        ax.set_title(f"{subset} | CTD={int(row['CTD_size'])}")
        ax.set_xticks(np.arange(len(MATRICES)), MATRICES)
        ax.set_xlabel("Projection matrix")
        ax.set_yticks(np.arange(0, num_layers, 5))
        ax.set_yticklabels([str(i) for i in range(0, num_layers, 5)])
        for layer in range(num_layers):
            for col in range(len(MATRICES)):
                value = int(data[layer, col])
                if value:
                    ax.text(col, layer, str(value), ha="center", va="center", fontsize=6, color="white")
    axes[0].set_ylabel("Layer")
    cbar = fig.colorbar(image, ax=axes, fraction=0.028, pad=0.025)
    cbar.set_label("CTD count")
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_summary_md(path: Path, summary_rows: List[Dict[str, Any]], share_rows: List[Dict[str, Any]]) -> None:
    lines = [
        "# Stage 7 Shared Neuron Discovery",
        "",
        "CTD is the exact A/B/C intersection of single-type TDNs by `(layer, matrix, index)`.",
        "",
        "## Shared Counts",
        "",
    ]
    for row in summary_rows:
        lines.append(
            f"- {row['subset']}: |CTD|={row['CTD_size']}, "
            f"|TDN_A|={row['TDN_sizes']['A']}, |TDN_B|={row['TDN_sizes']['B']}, |TDN_C|={row['TDN_sizes']['C']}"
        )
    lines.extend(["", "## Share Rates And Base Capability", ""])
    for row in share_rows:
        lines.append(
            f"- {row['subset']} / {row['task_type']}: ShareRate={row['share_rate']:.4f}, "
            f"Base ToolAcc={row['base_tool_accuracy']:.4f}, Base Acc={row['base_accuracy']:.4f}"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    single_type_root = Path(args.single_type_dir) if args.single_type_dir else data_root / "neurons" / args.model_alias / "single_type_by_subset"
    causal_root = Path(args.causal_dir) if args.causal_dir else data_root / "causal_validation" / args.model_alias / "single_type_by_subset"
    base_output = Path(args.output_dir) if args.output_dir else data_root / "neurons"
    output_root = base_output / args.model_alias / "shared_by_subset"
    vis_root = Path(args.visualization_dir) if args.visualization_dir else data_root / "visualizations" / args.model_alias / "shared_by_subset"
    if args.skip_existing and not args.overwrite and (output_root / "manifest.json").exists():
        print(f"[skip] stage 7 already complete: {output_root}")
        return
    ensure_empty_or_create(output_root, args.overwrite)
    if args.make_figures:
        ensure_empty_or_create(vis_root, args.overwrite)

    base_metrics = read_base_metrics(causal_root)
    summary_rows: List[Dict[str, Any]] = []
    all_share_rows: List[Dict[str, Any]] = []
    all_layer_rows: List[Dict[str, Any]] = []
    all_matrix_rows: List[Dict[str, Any]] = []
    for subset in args.subsets:
        summary, share_rows, layer_rows, matrix_rows = discover_subset(
            subset,
            single_type_root,
            output_root,
            base_metrics,
        )
        summary_rows.append(summary)
        all_share_rows.extend(share_rows)
        all_layer_rows.extend(layer_rows)
        all_matrix_rows.extend(matrix_rows)

    write_csv(
        output_root / "shared_summary.csv",
        [
            {
                "subset": row["subset"],
                "CTD_size": row["CTD_size"],
                "exact_CTD_size": row["exact_CTD_size"],
                "TDN_A_size": row["TDN_sizes"]["A"],
                "TDN_B_size": row["TDN_sizes"]["B"],
                "TDN_C_size": row["TDN_sizes"]["C"],
                "pairwise_AB_size": row["pairwise_sizes"]["AB"],
                "pairwise_AC_size": row["pairwise_sizes"]["AC"],
                "pairwise_BC_size": row["pairwise_sizes"]["BC"],
                "stage5_candidate_mode": row["stage5_candidate_universe"].get("mode", ""),
            }
            for row in summary_rows
        ],
        [
            "subset",
            "CTD_size",
            "exact_CTD_size",
            "TDN_A_size",
            "TDN_B_size",
            "TDN_C_size",
            "pairwise_AB_size",
            "pairwise_AC_size",
            "pairwise_BC_size",
            "stage5_candidate_mode",
        ],
    )
    write_csv(
        output_root / "share_rates.csv",
        all_share_rows,
        [
            "subset",
            "task_type",
            "TDN_size",
            "CTD_size",
            "exact_CTD_size",
            "share_rate",
            "base_accuracy",
            "base_tool_accuracy",
            "base_tool_call_rate",
        ],
    )
    write_csv(output_root / "layer_counts.csv", all_layer_rows, ["subset", "layer", "shared_count"])
    write_csv(output_root / "matrix_counts.csv", all_matrix_rows, ["subset", "matrix", "shared_count"])

    figures: List[str] = []
    if args.make_figures:
        fig4 = vis_root / "fig4_shared_abundance_vs_capability.png"
        heatmap = vis_root / "fig_shared_neuron_heatmap.png"
        plot_fig4_abundance(summary_rows, all_share_rows, fig4)
        plot_shared_heatmap(summary_rows, output_root, heatmap)
        type_figures: List[str] = []
        for subset in args.subsets:
            path = vis_root / f"fig4_share_rate_vs_type_capability_{subset}.png"
            plot_share_rate_by_type(subset, all_share_rows, path)
            type_figures.append(str(path))
        figures = [str(fig4), str(heatmap), *type_figures]
        write_summary_md(vis_root / "summary.md", summary_rows, all_share_rows)

    write_summary_md(output_root / "summary.md", summary_rows, all_share_rows)
    write_json(
        output_root / "manifest.json",
        {
            "stage": "stage7_shared_neuron_discovery",
            "model_alias": args.model_alias,
            "data_root": str(data_root),
            "single_type_dir": str(single_type_root),
            "causal_dir": str(causal_root),
            "output_dir": str(output_root),
            "visualization_dir": str(vis_root) if args.make_figures else "",
            "make_figures": args.make_figures,
            "subsets": list(args.subsets),
            "neuron_definition": "Exact (layer,matrix,index) identity; Q/K/V are projection rows and O is an output-projection column.",
            "shared_definition": "CTD_{m,l}=TDN_{m,A,l} intersect TDN_{m,B,l} intersect TDN_{m,C,l}; CTD_m=union_l CTD_{m,l}.",
            "figure_policy": "Discovery stage writes neuron sets only by default. Causal metric figures are generated after stage 8.",
            "subset_summaries": summary_rows,
            "figures": figures,
        },
    )
    if args.make_figures:
        write_json(
            vis_root / "manifest.json",
            {
                "stage": "stage7_shared_neuron_visualization",
                "model_alias": args.model_alias,
                "output_dir": str(vis_root),
                "figures": figures,
            },
        )
    print(f"saved stage-7 shared neurons: {output_root}")
    if args.make_figures:
        print(f"saved stage-7 visualizations: {vis_root}")


if __name__ == "__main__":
    main()
