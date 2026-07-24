#!/usr/bin/env python3
"""Stage 8 cross-type causal validation.

It validates cross-type shared neurons in the Who Transfers
Safety style:

1. Base
2. M-Random: random mask with the same layer/matrix shape as CTD_m
3. M-CTD: mask shared CTD_m
4. M-Private_c: mask Private_{m,c} = TDN_{m,c} \\ CTD_m

Metrics reuse the stage-6 When2Tool-aligned evaluator: Acc, TC, AvgTC, TCR,
ToolAcc, ToolNecessaryAcc, NoToolAcc, and OverCall. Cross-type aggregate rows
additionally report DeltaAcc, DeltaTCR, and VarAcc over A/B/C.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
STAGE6_DIR = CODE_ROOT / "05_single_type_causal_validation"
for path in (str(STAGE6_DIR), str(CODE_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import run_single_type_causal_validation as stage6  # noqa: E402
from common import causal_plots  # noqa: E402

TASK_TYPES = ("A", "B", "C")
SUBSETS = ("single_hop", "multi_hop")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-alias", default="qwen3-4b-instruct")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--modified-dir", required=True)
    parser.add_argument("--shared-dir", required=True, help="Stage-7 shared_by_subset dir.")
    parser.add_argument("--single-type-dir", required=True, help="Stage-5 single_type_by_subset dir, used for matrix dims.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-types", nargs="+", default=list(TASK_TYPES), choices=TASK_TYPES)
    parser.add_argument("--subsets", nargs="+", default=list(SUBSETS), choices=SUBSETS)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-tasks-per-type", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--enable-thinking", default="auto", choices=["auto", "true", "false"])
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--record-mode", default="lite", choices=["lite", "full", "off"])
    parser.add_argument(
        "--interventions",
        nargs="+",
        default=[],
        help="Optional worker filter. Use Base, M-Random, M-CTD, M-Private, or concrete names such as M-Private_A.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--refresh-existing",
        action="store_true",
        help="Recompute tables/manifests from existing per_task.jsonl files without loading the model.",
    )
    return parser.parse_args()


def intervention_selected(requested: List[str], task_type: str, intervention: str) -> bool:
    if not requested:
        return True
    aliases = {intervention}
    if intervention == f"M-Private_{task_type}":
        aliases.update({"M-Private", "M-Private_c"})
    return bool(set(requested) & aliases)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"Missing jsonl: {path}")
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


def variance(values: List[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def load_private_rows(shared_subset_dir: Path, task_type: str) -> List[Dict[str, Any]]:
    return read_jsonl(shared_subset_dir / f"private_{task_type}_neurons.jsonl")


EFFECT_COLUMNS = [
    "subset_scope",
    "intervention_group",
    "mean_accuracy",
    "mean_tool_accuracy",
    "mean_tool_necessary_accuracy",
    "mean_no_tool_accuracy",
    "delta_accuracy",
    "delta_tcr",
    "delta_tool_accuracy",
    "delta_tool_necessary_accuracy",
    "delta_no_tool_accuracy",
    "delta_overcall",
    "var_accuracy",
    "var_tcr",
    "total_tool_calls",
    "total_n",
]


def metric_value(row: Dict[str, Any], key: str) -> float:
    if key == "tool_necessary_accuracy":
        return float(row.get("tool_necessary_accuracy", row.get("recall_tool", 0.0)))
    if key == "no_tool_accuracy":
        if "no_tool_accuracy" in row:
            return float(row["no_tool_accuracy"])
        return 1.0 - float(row.get("overcall", 0.0))
    return float(row[key])


def effect_rows_for_subset(summary_rows: List[Dict[str, Any]], subset: str, task_types: List[str]) -> List[Dict[str, Any]]:
    by_task_intervention = {
        (row["task_type"], row["intervention"]): row
        for row in summary_rows
        if row.get("subset_scope") == subset
    }
    out: List[Dict[str, Any]] = []
    groups = [
        ("Base", {task_type: "Base" for task_type in task_types}),
        ("M-Random", {task_type: "M-Random" for task_type in task_types}),
        ("M-CTD", {task_type: "M-CTD" for task_type in task_types}),
        ("M-Private_c", {task_type: f"M-Private_{task_type}" for task_type in task_types}),
    ]
    for group_name, intervention_by_task in groups:
        delta_acc: List[float] = []
        delta_tcr: List[float] = []
        delta_tool_acc: List[float] = []
        delta_tool_necessary_acc: List[float] = []
        delta_no_tool_acc: List[float] = []
        delta_overcall: List[float] = []
        mean_acc: List[float] = []
        mean_tool_acc: List[float] = []
        mean_tool_necessary_acc: List[float] = []
        mean_no_tool_acc: List[float] = []
        total_tc = 0
        total_n = 0
        for task_type in task_types:
            base = by_task_intervention[(task_type, "Base")]
            row = by_task_intervention[(task_type, intervention_by_task[task_type])]
            delta_acc.append(float(row["accuracy"]) - float(base["accuracy"]))
            delta_tcr.append(float(row["tool_call_rate"]) - float(base["tool_call_rate"]))
            delta_tool_acc.append(float(row["tool_accuracy"]) - float(base["tool_accuracy"]))
            delta_tool_necessary_acc.append(metric_value(row, "tool_necessary_accuracy") - metric_value(base, "tool_necessary_accuracy"))
            delta_no_tool_acc.append(metric_value(row, "no_tool_accuracy") - metric_value(base, "no_tool_accuracy"))
            delta_overcall.append(float(row["overcall"]) - float(base["overcall"]))
            mean_acc.append(float(row["accuracy"]))
            mean_tool_acc.append(float(row["tool_accuracy"]))
            mean_tool_necessary_acc.append(metric_value(row, "tool_necessary_accuracy"))
            mean_no_tool_acc.append(metric_value(row, "no_tool_accuracy"))
            total_tc += int(row["tool_calls"])
            total_n += int(row["n"])
        out.append(
            {
                "subset_scope": subset,
                "intervention_group": group_name,
                "mean_accuracy": sum(mean_acc) / len(mean_acc),
                "mean_tool_accuracy": sum(mean_tool_acc) / len(mean_tool_acc),
                "mean_tool_necessary_accuracy": sum(mean_tool_necessary_acc) / len(mean_tool_necessary_acc),
                "mean_no_tool_accuracy": sum(mean_no_tool_acc) / len(mean_no_tool_acc),
                "delta_accuracy": sum(delta_acc) / len(delta_acc),
                "delta_tcr": sum(delta_tcr) / len(delta_tcr),
                "delta_tool_accuracy": sum(delta_tool_acc) / len(delta_tool_acc),
                "delta_tool_necessary_accuracy": sum(delta_tool_necessary_acc) / len(delta_tool_necessary_acc),
                "delta_no_tool_accuracy": sum(delta_no_tool_acc) / len(delta_no_tool_acc),
                "delta_overcall": sum(delta_overcall) / len(delta_overcall),
                "var_accuracy": variance(delta_acc),
                "var_tcr": variance(delta_tcr),
                "total_tool_calls": total_tc,
                "total_n": total_n,
            }
        )
    return out


def markdown_table(summary_rows: List[Dict[str, Any]], effect_rows: List[Dict[str, Any]]) -> str:
    lines = [
        "# Stage 8 Cross-Type Causal Validation",
        "",
        "## When2Tool-Aligned Metrics",
        "",
        "| Model | Subset | Type | Intervention | MaskN | N | Acc | TC | AvgTC | TCR | ToolAcc | ToolNecessaryAcc | NoToolAcc | OverCall |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['model_alias']} | {row['subset_scope']} | {row['task_type']} | {row['intervention']} | "
            f"{row['mask_count']} | {row['n']} | {row['accuracy']:.4f} | {row['tool_calls']} | "
            f"{row['avg_tool_calls']:.4f} | {row['tool_call_rate']:.4f} | {row['tool_accuracy']:.4f} | "
            f"{metric_value(row, 'tool_necessary_accuracy'):.4f} | {metric_value(row, 'no_tool_accuracy'):.4f} | "
            f"{row['overcall']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Cross-Type Effects",
            "",
            "| Subset | Intervention | MeanAcc | MeanToolAcc | MeanToolNecessaryAcc | MeanNoToolAcc | DeltaAcc | DeltaTCR | DeltaToolAcc | DeltaToolNecessaryAcc | DeltaNoToolAcc | VarAcc | VarTCR | TC | N |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in effect_rows:
        lines.append(
            f"| {row['subset_scope']} | {row['intervention_group']} | {row['mean_accuracy']:.4f} | "
            f"{row['mean_tool_accuracy']:.4f} | {row['mean_tool_necessary_accuracy']:.4f} | "
            f"{row['mean_no_tool_accuracy']:.4f} | {row['delta_accuracy']:.4f} | {row['delta_tcr']:.4f} | "
            f"{row['delta_tool_accuracy']:.4f} | {row['delta_tool_necessary_accuracy']:.4f} | "
            f"{row['delta_no_tool_accuracy']:.4f} | {row['var_accuracy']:.6f} | {row['var_tcr']:.6f} | "
            f"{row['total_tool_calls']} | {row['total_n']} |"
        )
    lines.extend(
        [
            "",
            "Metric notes:",
            "- Acc / TC / AvgTC / TCR follow When2Tool-style final-answer and tool-call metrics.",
            "- ToolAcc is tool-decision accuracy: whether used_tool equals tool_necessary.",
            "- ToolNecessaryAcc is accuracy on tool_necessary=1 samples; NoToolAcc is accuracy on tool_necessary=0 samples.",
            "- DeltaAcc and DeltaTCR are averaged over A/B/C against Base.",
            "- VarAcc is the variance of per-type accuracy deltas across A/B/C, following the cross-type consistency check.",
            "",
        ]
    )
    return "\n".join(lines)


def run_subset(
    args: argparse.Namespace,
    subset: str,
    generator: stage6.CausalHFGenerator,
    tool_format: str,
    output_root: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    shared_subset_dir = Path(args.shared_dir) / subset
    single_type_subset_dir = Path(args.single_type_dir) / subset
    subset_output = output_root / subset
    subset_output.mkdir(parents=True, exist_ok=True)

    shared_manifest = read_json(shared_subset_dir / "manifest.json")
    ctd_rows = read_jsonl(shared_subset_dir / "CTD_neurons.jsonl")
    ctd_mask = stage6.rows_to_mask(ctd_rows)
    dims = stage6.load_matrix_dims(single_type_subset_dir)
    random_mask = stage6.random_like_mask(ctd_rows, dims, args.seed, f"{subset}:CTD")
    private_masks = {
        task_type: stage6.rows_to_mask(load_private_rows(shared_subset_dir, task_type))
        for task_type in args.task_types
    }

    tasks_by_type = stage6.load_test_tasks(
        Path(args.modified_dir),
        [subset],
        args.split,
        args.task_types,
        args.max_tasks_per_type,
    )
    for task_type, tasks in tasks_by_type.items():
        print(f"{subset}/{task_type}: loaded {len(tasks)} {args.split} tasks")
        print(f"{subset}/{task_type}: envs={sorted({task['env_name'] for task in tasks})}")

    all_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for task_type in args.task_types:
        runs = [
            ("Base", None),
            ("M-Random", random_mask),
            ("M-CTD", ctd_mask),
            (f"M-Private_{task_type}", private_masks[task_type]),
        ]
        for intervention, mask in runs:
            if not intervention_selected(args.interventions, task_type, intervention):
                continue
            rows, summary = stage6.run_one(
                generator,
                tasks_by_type[task_type],
                args.model_alias,
                subset,
                task_type,
                intervention,
                mask,
                args.max_rounds,
                tool_format,
                args.record_mode,
                subset_output,
            )
            all_rows.extend(rows)
            summary_rows.append(summary)

    effect_rows = effect_rows_for_subset(summary_rows, subset, list(args.task_types))
    write_csv(subset_output / "summary_table.csv", summary_rows, stage6.SUMMARY_COLUMNS)
    write_csv(subset_output / "cross_type_effects.csv", effect_rows, EFFECT_COLUMNS)
    stage6.write_jsonl(subset_output / "all_per_task.jsonl", all_rows)
    (subset_output / "summary.md").write_text(markdown_table(summary_rows, effect_rows), encoding="utf-8")
    figures = causal_plots.plot_stage8_subset(
        subset_output,
        shared_subset_dir,
        single_type_subset_dir,
        summary_rows,
        effect_rows,
    )
    stage6.write_json(
        subset_output / "manifest.json",
        {
            "stage": "stage8_cross_type_causal_validation_scope",
            "model_alias": args.model_alias,
            "subset_scope": subset,
            "shared_dir": str(shared_subset_dir),
            "single_type_dir": str(single_type_subset_dir),
            "output_dir": str(subset_output),
            "split": args.split,
            "task_types": list(args.task_types),
            "ctd_size": len(ctd_rows),
            "exact_CTD_size": int(shared_manifest.get("exact_CTD_size", len(ctd_rows))),
            "neuron_definition": "Exact (layer,matrix,index) identity; Q/K/V are projection rows and O is an output-projection column.",
            "interventions": ["Base", "M-Random", "M-CTD", "M-Private_c"],
            "private_definition": "Private_{m,c}=TDN_{m,c} minus CTD_m.",
            "mask_implementation": "Q/K/V zero projection-output coordinates; O zero o_proj input coordinates, equivalent to masking W_O columns.",
            "metrics": {
                "Acc": "final-answer accuracy, aligned with When2Tool",
                "TC": "total tool calls",
                "AvgTC": "average tool calls per task",
                "TCR": "tool calls divided by expected steps",
                "ToolAcc": "mean 1[used_tool == tool_necessary]",
                "ToolNecessaryAcc": "mean used_tool on tool_necessary=1 samples",
                "NoToolAcc": "mean 1[not used_tool] on tool_necessary=0 samples",
                "DeltaAcc": "mean over A/B/C of Acc(intervention,c)-Acc(Base,c)",
                "DeltaTCR": "mean over A/B/C of TCR(intervention,c)-TCR(Base,c)",
                "VarAcc": "variance over A/B/C of Acc deltas",
            },
            "figures": figures,
            "summary_rows": summary_rows,
            "effect_rows": effect_rows,
        },
    )
    return all_rows, summary_rows


def refresh_subset(
    args: argparse.Namespace,
    subset: str,
    output_root: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    shared_subset_dir = Path(args.shared_dir) / subset
    single_type_subset_dir = Path(args.single_type_dir) / subset
    subset_output = output_root / subset
    if not subset_output.exists():
        raise FileNotFoundError(f"Missing stage-8 subset output: {subset_output}")

    shared_manifest = read_json(shared_subset_dir / "manifest.json")
    ctd_rows = read_jsonl(shared_subset_dir / "CTD_neurons.jsonl")
    all_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for task_type in args.task_types:
        interventions = ["Base", "M-Random", "M-CTD", f"M-Private_{task_type}"]
        for intervention in interventions:
            if not intervention_selected(args.interventions, task_type, intervention):
                continue
            run_dir = subset_output / task_type / intervention
            rows = read_jsonl(run_dir / "per_task.jsonl")
            summary = stage6.summarize_rows(rows)
            old_summary = read_json(run_dir / "summary.json") if (run_dir / "summary.json").exists() else {}
            summary.update(
                {
                    "model_alias": old_summary.get("model_alias", args.model_alias),
                    "subset_scope": old_summary.get("subset_scope", subset),
                    "task_type": old_summary.get("task_type", task_type),
                    "intervention": old_summary.get("intervention", intervention),
                    "mask_count": old_summary.get("mask_count", rows[0].get("mask_count", 0) if rows else 0),
                }
            )
            stage6.write_json(run_dir / "summary.json", summary)
            all_rows.extend(rows)
            summary_rows.append(summary)

    effect_rows = effect_rows_for_subset(summary_rows, subset, list(args.task_types))
    write_csv(subset_output / "summary_table.csv", summary_rows, stage6.SUMMARY_COLUMNS)
    write_csv(subset_output / "cross_type_effects.csv", effect_rows, EFFECT_COLUMNS)
    stage6.write_jsonl(subset_output / "all_per_task.jsonl", all_rows)
    (subset_output / "summary.md").write_text(markdown_table(summary_rows, effect_rows), encoding="utf-8")
    figures = causal_plots.plot_stage8_subset(
        subset_output,
        shared_subset_dir,
        single_type_subset_dir,
        summary_rows,
        effect_rows,
    )
    stage6.write_json(
        subset_output / "manifest.json",
        {
            "stage": "stage8_cross_type_causal_validation_scope",
            "model_alias": args.model_alias,
            "subset_scope": subset,
            "shared_dir": str(shared_subset_dir),
            "single_type_dir": str(single_type_subset_dir),
            "output_dir": str(subset_output),
            "split": args.split,
            "task_types": list(args.task_types),
            "ctd_size": len(ctd_rows),
            "exact_CTD_size": int(shared_manifest.get("exact_CTD_size", len(ctd_rows))),
            "neuron_definition": "Exact (layer,matrix,index) identity; Q/K/V are projection rows and O is an output-projection column.",
            "interventions": ["Base", "M-Random", "M-CTD", "M-Private_c"],
            "private_definition": "Private_{m,c}=TDN_{m,c} minus CTD_m.",
            "mask_implementation": "Q/K/V zero projection-output coordinates; O zero o_proj input coordinates, equivalent to masking W_O columns.",
            "metrics": {
                "Acc": "final-answer accuracy, aligned with When2Tool",
                "TC": "total tool calls",
                "AvgTC": "average tool calls per task",
                "TCR": "tool calls divided by expected steps",
                "ToolAcc": "mean 1[used_tool == tool_necessary]",
                "ToolNecessaryAcc": "mean used_tool on tool_necessary=1 samples",
                "NoToolAcc": "mean 1[not used_tool] on tool_necessary=0 samples",
                "DeltaAcc": "mean over A/B/C of Acc(intervention,c)-Acc(Base,c)",
                "DeltaTCR": "mean over A/B/C of TCR(intervention,c)-TCR(Base,c)",
                "VarAcc": "variance over A/B/C of Acc deltas",
            },
            "figures": figures,
            "summary_rows": summary_rows,
            "effect_rows": effect_rows,
        },
    )
    return all_rows, summary_rows


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir) / args.model_alias / "cross_type_by_subset"
    if output_root.exists() and not (args.overwrite or args.refresh_existing):
        raise FileExistsError(f"Output exists: {output_root}. Use --overwrite.")
    output_root.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    effect_rows: List[Dict[str, Any]] = []
    if args.refresh_existing:
        for subset in args.subsets:
            subset_rows, subset_summary = refresh_subset(args, subset, output_root)
            all_rows.extend(subset_rows)
            summary_rows.extend(subset_summary)
            effect_rows.extend(effect_rows_for_subset(subset_summary, subset, list(args.task_types)))
    else:
        tool_format = stage6.detect_tool_format(args.model_path)
        generator = stage6.CausalHFGenerator(args.model_path, args)
        for subset in args.subsets:
            subset_rows, subset_summary = run_subset(args, subset, generator, tool_format, output_root)
            all_rows.extend(subset_rows)
            summary_rows.extend(subset_summary)
            effect_rows.extend(effect_rows_for_subset(subset_summary, subset, list(args.task_types)))

    write_csv(output_root / "summary_table.csv", summary_rows, stage6.SUMMARY_COLUMNS)
    write_csv(output_root / "cross_type_effects.csv", effect_rows, EFFECT_COLUMNS)
    stage6.write_jsonl(output_root / "all_per_task.jsonl", all_rows)
    (output_root / "summary.md").write_text(markdown_table(summary_rows, effect_rows), encoding="utf-8")
    figures = causal_plots.plot_stage8_overview(output_root, effect_rows)
    stage6.write_json(
        output_root / "manifest.json",
        {
            "stage": "stage8_cross_type_causal_validation",
            "model_alias": args.model_alias,
            "model_path": args.model_path,
            "modified_dir": args.modified_dir,
            "shared_dir": args.shared_dir,
            "single_type_dir": args.single_type_dir,
            "output_dir": str(output_root),
            "split": args.split,
            "subsets": list(args.subsets),
            "task_types": list(args.task_types),
            "prompt_mode": "current",
            "reasoning_mode": "no_reasoning",
            "neuron_definition": "Exact (layer,matrix,index) identity; Q/K/V are projection rows and O is an output-projection column.",
            "shared_definition": "CTD_m is the exact A/B/C shared neuron set from stage 7.",
            "interventions": ["Base", "M-Random", "M-CTD", "M-Private_c"],
            "metrics": {
                "Acc": "final-answer accuracy, aligned with When2Tool",
                "TC": "total tool calls",
                "AvgTC": "average tool calls per task",
                "TCR": "tool calls divided by expected steps",
                "ToolAcc": "mean 1[used_tool == tool_necessary]",
                "ToolNecessaryAcc": "mean used_tool on tool_necessary=1 samples",
                "NoToolAcc": "mean 1[not used_tool] on tool_necessary=0 samples",
                "DeltaAcc": "mean over A/B/C of Acc(intervention,c)-Acc(Base,c)",
                "DeltaTCR": "mean over A/B/C of TCR(intervention,c)-TCR(Base,c)",
                "VarAcc": "variance over A/B/C of Acc deltas",
            },
            "figures": figures,
            "summary_rows": summary_rows,
            "effect_rows": effect_rows,
        },
    )
    print(f"saved stage-8 cross-type causal validation: {output_root}")


if __name__ == "__main__":
    main()
