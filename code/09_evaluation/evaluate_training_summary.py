#!/usr/bin/env python3
"""Stage 10 evaluation and summary after CTD neuron training.

It evaluates the stage-9 CTD-training delta checkpoint with
the same When2Tool-style evaluator used in causal validation, then summarizes
Default vs CTD-training.

Neuron definition:
- Q/K/V neurons are rows of W_Q/W_K/W_V.
- O neurons are columns of W_O.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
STAGE6_DIR = CODE_ROOT / "05_single_type_causal_validation"
STAGE9_DIR = CODE_ROOT / "08_training"
for path in (str(STAGE6_DIR), str(STAGE9_DIR), str(CODE_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import run_single_type_causal_validation as stage6  # noqa: E402
import train_shared_neurons as stage9  # noqa: E402
from common.io_utils import write_json, write_jsonl  # noqa: E402

TASK_TYPES = ("A", "B", "C")
SUBSETS = ("single_hop", "multi_hop")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-alias", default="qwen3-4b-instruct")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--modified-dir", required=True)
    parser.add_argument("--training-dir", required=True, help="Stage-9 neuron_training_by_subset dir.")
    parser.add_argument("--default-eval-dir", required=True, help="Stage-8 cross_type_by_subset dir containing Base per-task rows.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-types", nargs="+", default=list(TASK_TYPES), choices=TASK_TYPES)
    parser.add_argument("--subsets", nargs="+", default=list(SUBSETS), choices=SUBSETS)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-eval-tasks-per-type", type=int, default=0)
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--enable-thinking", default="false", choices=["auto", "true", "false"])
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--record-mode", default="lite", choices=["lite", "full", "off"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--refresh-existing",
        action="store_true",
        help="Recompute tables/manifests from existing CTD-training per_task.jsonl files without loading the model.",
    )
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


def metric_value(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    if key == "tool_necessary_accuracy":
        return float(row.get("tool_necessary_accuracy", row.get("recall_tool", default)))
    if key == "no_tool_accuracy":
        if "no_tool_accuracy" in row:
            return float(row["no_tool_accuracy"])
        return 1.0 - float(row.get("overcall", default))
    value = row.get(key, default)
    if value in {"", None, "NA"}:
        return default
    return float(value)


def apply_ctd_delta(model: Any, checkpoint_path: Path) -> Dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    applied: List[Dict[str, Any]] = []
    with torch.no_grad():
        for item in ckpt["trainable_neuron_deltas"]:
            layer = int(item["layer"])
            matrix = str(item["matrix"])
            index = int(item["index"])
            _param_name, module, orientation = stage9.module_for_neuron(model, layer, matrix)
            delta = item["delta"].to(device=module.weight.device, dtype=module.weight.dtype)
            if orientation == "row":
                module.weight[index, :] += delta
            elif orientation == "column":
                module.weight[:, index] += delta
            else:
                raise ValueError(f"Unknown orientation: {orientation}")
            applied.append(
                {
                    "neuron_id": item.get("neuron_id", f"L{layer:02d}.{matrix}.{index:05d}"),
                    "layer": layer,
                    "matrix": matrix,
                    "index": index,
                    "orientation": orientation,
                    "delta_shape": list(delta.shape),
                    "delta_abs_max": float(delta.detach().float().abs().max().item()) if delta.numel() else 0.0,
                }
            )
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_stage": ckpt.get("stage"),
        "neuron_definition": ckpt.get("neuron_definition"),
        "applied_neurons": applied,
    }


def load_default_rows(default_eval_dir: Path, subset: str, task_types: Iterable[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    all_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for task_type in task_types:
        run_dir = default_eval_dir / subset / task_type / "Base"
        rows = read_jsonl(run_dir / "per_task.jsonl")
        for row in rows:
            row["method"] = "Default"
            row["intervention"] = "Default"
            row["subset_scope"] = subset
        summary = stage6.summarize_rows(rows)
        summary.update(
            {
                "model_alias": rows[0].get("model_alias") if rows else "",
                "subset_scope": subset,
                "task_type": task_type,
                "method": "Default",
                "intervention": "Default",
                "mask_count": 0,
                "auroc": "NA",
            }
        )
        all_rows.extend(rows)
        summary_rows.append(summary)
    return all_rows, summary_rows


def run_trained_eval(
    args: argparse.Namespace,
    subset: str,
    generator: stage6.CausalHFGenerator,
    tool_format: str,
    output_root: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    tasks_by_type = stage6.load_test_tasks(
        Path(args.modified_dir),
        [subset],
        args.split,
        args.task_types,
        args.max_eval_tasks_per_type,
    )
    all_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for task_type in args.task_types:
        tasks = tasks_by_type[task_type]
        print(f"{subset}/{task_type}/CTD-training: loaded {len(tasks)} {args.split} tasks")
        outputs = stage6.evaluate_tasks(tasks, generator, None, args.max_rounds, tool_format, args.record_mode)
        run_dir = output_root / subset / task_type / "CTD-training"
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "outputs.json", outputs)
        rows = [
            stage6.per_task_row(item, args.model_alias, task_type, "CTD-training", 0)
            for item in outputs
        ]
        for row in rows:
            row["method"] = "CTD-training"
            row["subset_scope"] = subset
        write_jsonl(run_dir / "per_task.jsonl", rows)
        summary = stage6.summarize_rows(rows)
        summary.update(
            {
                "model_alias": args.model_alias,
                "subset_scope": subset,
                "task_type": task_type,
                "method": "CTD-training",
                "intervention": "CTD-training",
                "mask_count": 0,
                "auroc": "NA",
            }
        )
        write_json(run_dir / "summary.json", summary)
        all_rows.extend(rows)
        summary_rows.append(summary)
    return all_rows, summary_rows


def refresh_trained_eval(
    args: argparse.Namespace,
    subset: str,
    output_root: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    all_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for task_type in args.task_types:
        run_dir = output_root / subset / task_type / "CTD-training"
        rows = read_jsonl(run_dir / "per_task.jsonl")
        for row in rows:
            row["method"] = "CTD-training"
            row["intervention"] = "CTD-training"
            row["subset_scope"] = subset
        summary = stage6.summarize_rows(rows)
        old_summary = read_json(run_dir / "summary.json") if (run_dir / "summary.json").exists() else {}
        summary.update(
            {
                "model_alias": old_summary.get("model_alias", args.model_alias),
                "subset_scope": subset,
                "task_type": task_type,
                "method": "CTD-training",
                "intervention": "CTD-training",
                "mask_count": old_summary.get("mask_count", 0),
                "auroc": "NA",
            }
        )
        write_json(run_dir / "summary.json", summary)
        all_rows.extend(rows)
        summary_rows.append(summary)
    return all_rows, summary_rows


def aggregate_rows(rows: List[Dict[str, Any]], model_alias: str, subset_scope: str, method: str, task_type: str = "ALL") -> Dict[str, Any]:
    summary = stage6.summarize_rows(rows)
    summary.update(
        {
            "model_alias": model_alias,
            "subset_scope": subset_scope,
            "task_type": task_type,
            "method": method,
            "intervention": method,
            "mask_count": 0,
            "auroc": "NA",
        }
    )
    return summary


def comparison_delta_rows(summary_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_key = {
        (row["subset_scope"], row["task_type"], row["method"]): row
        for row in summary_rows
    }
    out: List[Dict[str, Any]] = []
    for subset_scope, task_type, method in sorted(by_key):
        if method != "CTD-training":
            continue
        trained = by_key[(subset_scope, task_type, method)]
        default = by_key.get((subset_scope, task_type, "Default"))
        if default is None:
            continue
        delta_acc = metric_value(trained, "accuracy") - metric_value(default, "accuracy")
        delta_avg_tc = metric_value(trained, "avg_tool_calls") - metric_value(default, "avg_tool_calls")
        cost: Any = "NA"
        if delta_avg_tc < 0:
            cost = delta_acc / (-delta_avg_tc)
        out.append(
            {
                "model_alias": trained["model_alias"],
                "subset_scope": subset_scope,
                "task_type": task_type,
                "method": "CTD-training",
                "baseline": "Default",
                "delta_accuracy": delta_acc,
                "delta_tc": int(trained["tool_calls"]) - int(default["tool_calls"]),
                "delta_avg_tool_calls": delta_avg_tc,
                "delta_tcr": metric_value(trained, "tool_call_rate") - metric_value(default, "tool_call_rate"),
                "delta_tool_accuracy": metric_value(trained, "tool_accuracy") - metric_value(default, "tool_accuracy"),
                "delta_tool_necessary_accuracy": metric_value(trained, "tool_necessary_accuracy") - metric_value(default, "tool_necessary_accuracy"),
                "delta_no_tool_accuracy": metric_value(trained, "no_tool_accuracy") - metric_value(default, "no_tool_accuracy"),
                "cost_accuracy_per_saved_call": cost,
            }
        )
    return out


SUMMARY_COLUMNS = [
    "model_alias",
    "subset_scope",
    "task_type",
    "method",
    "n",
    "correct",
    "accuracy",
    "tool_calls",
    "avg_tool_calls",
    "tool_call_rate",
    "expected_steps",
    "tool_decision_correct",
    "tool_accuracy",
    "tool_necessary_correct",
    "tool_necessary_accuracy",
    "no_tool_correct",
    "no_tool_accuracy",
    "overcall",
    "auroc",
    "n_tool_necessary_1",
    "n_tool_necessary_0",
]

DELTA_COLUMNS = [
    "model_alias",
    "subset_scope",
    "task_type",
    "method",
    "baseline",
    "delta_accuracy",
    "delta_tc",
    "delta_avg_tool_calls",
    "delta_tcr",
    "delta_tool_accuracy",
    "delta_tool_necessary_accuracy",
    "delta_no_tool_accuracy",
    "cost_accuracy_per_saved_call",
]


def markdown_summary(summary_rows: List[Dict[str, Any]], delta_rows: List[Dict[str, Any]], training_manifests: Dict[str, Dict[str, Any]]) -> str:
    lines = [
        "# Stage 10 Evaluation And Summary",
        "",
        "## Training Before/After Metrics",
        "",
        "| Subset | Type | Method | N | Acc | TC | AvgTC | TCR | ToolAcc | ToolNecessaryAcc | NoToolAcc | AUROC |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        if row["task_type"] != "ALL":
            continue
        lines.append(
            f"| {row['subset_scope']} | {row['task_type']} | {row['method']} | {row['n']} | "
            f"{metric_value(row, 'accuracy'):.4f} | {row['tool_calls']} | {metric_value(row, 'avg_tool_calls'):.4f} | "
            f"{metric_value(row, 'tool_call_rate'):.4f} | {metric_value(row, 'tool_accuracy'):.4f} | "
            f"{metric_value(row, 'tool_necessary_accuracy'):.4f} | {metric_value(row, 'no_tool_accuracy'):.4f} | {row.get('auroc', 'NA')} |"
        )
    lines.extend(
        [
            "",
            "## Delta Against Default",
            "",
            "| Subset | Type | DeltaAcc | DeltaAvgTC | DeltaTCR | DeltaToolAcc | DeltaToolNecessaryAcc | DeltaNoToolAcc | Cost |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in delta_rows:
        if row["task_type"] != "ALL":
            continue
        cost = row["cost_accuracy_per_saved_call"]
        cost_text = "NA" if cost == "NA" else f"{float(cost):.4f}"
        lines.append(
            f"| {row['subset_scope']} | {row['task_type']} | {float(row['delta_accuracy']):.4f} | "
            f"{float(row['delta_avg_tool_calls']):.4f} | {float(row['delta_tcr']):.4f} | "
            f"{float(row['delta_tool_accuracy']):.4f} | {float(row['delta_tool_necessary_accuracy']):.4f} | "
            f"{float(row['delta_no_tool_accuracy']):.4f} | {cost_text} |"
        )
    lines.extend(["", "## Notes", ""])
    lines.append("- Default metrics are loaded from the existing stage-8 Base evaluation.")
    lines.append("- CTD-training metrics are generated by applying the stage-9 CTD delta checkpoint to the base model and running the same evaluator.")
    lines.append("- AUROC is kept as `NA` because no post-training probe score is produced here.")
    for subset, manifest in training_manifests.items():
        lines.append(
            f"- {subset}: CTD size={manifest.get('ctd_size')}, exact={manifest.get('ctd_exact_size')}."
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir) / args.model_alias / "training_summary_by_subset"
    if output_root.exists() and not (args.overwrite or args.refresh_existing):
        raise FileExistsError(f"Output exists: {output_root}. Use --overwrite.")
    if output_root.exists() and args.overwrite and not args.refresh_existing:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    tool_format = stage6.detect_tool_format(args.model_path)

    all_per_task_rows: List[Dict[str, Any]] = []
    type_summary_rows: List[Dict[str, Any]] = []
    aggregate_summary_rows: List[Dict[str, Any]] = []
    applied_delta_summaries: Dict[str, Dict[str, Any]] = {}
    training_manifests: Dict[str, Dict[str, Any]] = {}

    for subset in args.subsets:
        training_subset_dir = Path(args.training_dir) / subset
        training_manifest = read_json(training_subset_dir / "manifest.json")
        training_manifests[subset] = training_manifest
        if args.refresh_existing:
            applied_delta_summaries[subset] = {
                "checkpoint_path": str(training_subset_dir / "ctd_neuron_delta.pt"),
                "refreshed_from_existing_per_task": True,
            }
            generator = None
        else:
            generator = stage6.CausalHFGenerator(args.model_path, args)
            applied_delta_summaries[subset] = apply_ctd_delta(generator.model, training_subset_dir / "ctd_neuron_delta.pt")

        default_rows, default_summary = load_default_rows(Path(args.default_eval_dir), subset, args.task_types)
        if args.refresh_existing:
            trained_rows, trained_summary = refresh_trained_eval(args, subset, output_root)
        else:
            assert generator is not None
            trained_rows, trained_summary = run_trained_eval(args, subset, generator, tool_format, output_root)

        for row in default_rows + trained_rows:
            row["subset_scope"] = subset
        all_per_task_rows.extend(default_rows)
        all_per_task_rows.extend(trained_rows)
        type_summary_rows.extend(default_summary)
        type_summary_rows.extend(trained_summary)

        aggregate_summary_rows.append(aggregate_rows(default_rows, args.model_alias, subset, "Default"))
        aggregate_summary_rows.append(aggregate_rows(trained_rows, args.model_alias, subset, "CTD-training"))
        if generator is not None:
            del generator
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    for method in ("Default", "CTD-training"):
        rows = [row for row in all_per_task_rows if row.get("method") == method]
        aggregate_summary_rows.append(aggregate_rows(rows, args.model_alias, "ALL", method))

    summary_rows = type_summary_rows + aggregate_summary_rows
    delta_rows = comparison_delta_rows(summary_rows)

    write_csv(output_root / "training_comparison_by_type.csv", type_summary_rows, SUMMARY_COLUMNS)
    write_csv(output_root / "training_comparison_summary.csv", summary_rows, SUMMARY_COLUMNS)
    write_csv(output_root / "training_delta_summary.csv", delta_rows, DELTA_COLUMNS)
    write_jsonl(output_root / "all_per_task.jsonl", all_per_task_rows)
    write_json(output_root / "applied_delta_summary.json", applied_delta_summaries)
    (output_root / "summary.md").write_text(markdown_summary(summary_rows, delta_rows, training_manifests), encoding="utf-8")
    write_json(
        output_root / "manifest.json",
        {
            "stage": "stage10_evaluation_summary",
            "model_alias": args.model_alias,
            "model_path": args.model_path,
            "modified_dir": args.modified_dir,
            "training_dir": args.training_dir,
            "default_eval_dir": args.default_eval_dir,
            "output_dir": str(output_root),
            "subsets": list(args.subsets),
            "split": args.split,
            "task_types": list(args.task_types),
            "metrics": {
                "Acc": "final-answer accuracy",
                "TC": "total tool calls",
                "AvgTC": "average tool calls per task",
                "TCR": "tool calls divided by expected steps",
                "ToolAcc": "mean 1[used_tool == tool_necessary]",
                "ToolNecessaryAcc": "accuracy on tool_necessary=1 samples",
                "NoToolAcc": "accuracy on tool_necessary=0 samples",
                "OverCall": "tool-call rate on tool_necessary=0 samples",
                "AUROC": "NA in this stage; no post-training probe score is produced",
                "Cost": "DeltaAcc / -DeltaAvgTC when DeltaAvgTC < 0",
            },
            "neuron_definition": "Q/K/V neurons are W_Q/W_K/W_V rows; O neurons are W_O columns.",
            "refreshed_from_existing_per_task": bool(args.refresh_existing),
            "applied_delta_summary": applied_delta_summaries,
            "training_manifests": training_manifests,
        },
    )
    print(f"saved stage-10 evaluation summary: {output_root}")


if __name__ == "__main__":
    main()
