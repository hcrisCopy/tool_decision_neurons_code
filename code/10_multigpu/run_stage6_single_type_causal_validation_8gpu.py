#!/usr/bin/env python3
"""Single-node multi-GPU runner for stage 6 single-type causal validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common import multigpu_utils as mgpu  # noqa: E402

SUBSETS = ("single_hop", "multi_hop")
TASK_TYPES = ("A", "B", "C")
BASE_INTERVENTIONS = ("Base", "M-Random")
RUNNER_VERSION = "stage6_intervention_parallel_v1"


def interventions_for(task_type: str) -> List[str]:
    return [*BASE_INTERVENTIONS, f"M-TDN_{task_type}"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-alias", default="qwen3-4b-instruct")
    parser.add_argument("--model-path", default="", help="Optional. Defaults to configs/models.yaml by --model-alias.")
    parser.add_argument("--models-config", default=None)
    parser.add_argument("--allow-remote-model-download", action="store_true")
    parser.add_argument("--modified-dir", required=True)
    parser.add_argument("--neuron-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--subsets", nargs="+", default=list(SUBSETS), choices=SUBSETS)
    parser.add_argument("--task-types", nargs="+", default=list(TASK_TYPES), choices=TASK_TYPES)
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--cuda-devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-tasks-per-type", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--enable-thinking", default="false", choices=["model", "auto", "true", "false"])
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--record-mode", default="lite", choices=["lite", "full", "off"])
    parser.add_argument("--keep-workdir", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def final_root(args: argparse.Namespace) -> Path:
    return Path(args.output_dir) / args.model_alias / "single_type_by_subset"


def dependency_hash(args: argparse.Namespace, subset: str) -> str:
    path = Path(args.neuron_dir) / subset / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing stage5 subset manifest: {path}")
    return mgpu.file_sha256(path)


def expected_unit_meta(args: argparse.Namespace, subset: str, task_type: str, intervention: str) -> Dict[str, Any]:
    return {
        "runner_version": RUNNER_VERSION,
        "model_alias": args.model_alias,
        "model_path": args.model_path,
        "modified_dir": args.modified_dir,
        "neuron_dir": args.neuron_dir,
        "neuron_subset_manifest_sha256": dependency_hash(args, subset),
        "subset": subset,
        "task_type": task_type,
        "intervention": intervention,
        "split": args.split,
        "max_rounds": int(args.max_rounds),
        "max_new_tokens": int(args.max_new_tokens),
        "max_tasks_per_type": int(args.max_tasks_per_type),
        "seed": int(args.seed),
        "record_mode": args.record_mode,
    }


def unit_meta_matches(args: argparse.Namespace, subset: str, task_type: str, intervention: str) -> bool:
    path = final_root(args) / subset / task_type / intervention / "runner_meta.json"
    if not path.exists():
        return False
    try:
        old_meta = mgpu.read_json(path)
    except (OSError, ValueError):
        return False
    return old_meta == expected_unit_meta(args, subset, task_type, intervention)


def task_complete(args: argparse.Namespace, subset: str, task_type: str) -> bool:
    return all(unit_complete(args, subset, task_type, intervention) for intervention in interventions_for(task_type))


def unit_complete(args: argparse.Namespace, subset: str, task_type: str, intervention: str) -> bool:
    run_dir = final_root(args) / subset / task_type / intervention
    return mgpu.complete(run_dir, ("per_task.jsonl", "summary.json")) and unit_meta_matches(args, subset, task_type, intervention)


def stage_complete(args: argparse.Namespace) -> bool:
    return (
        (final_root(args) / "manifest.json").exists()
        and all(task_complete(args, subset, task_type) for subset in args.subsets for task_type in args.task_types)
    )


def safe_intervention(intervention: str) -> str:
    return intervention.replace("/", "_")


def unit_root(args: argparse.Namespace, subset: str, task_type: str, intervention: str) -> Path:
    return final_root(args) / "_stage6_workers" / subset / task_type / safe_intervention(intervention)


def build_job(args: argparse.Namespace, subset: str, task_type: str, intervention: str, device: str) -> mgpu.Job:
    temp_root = unit_root(args, subset, task_type, intervention)
    mgpu.remove_if_exists(temp_root)
    cmd = mgpu.command(
        "code/05_single_type_causal_validation/run_single_type_causal_validation.py",
        "--model-alias",
        args.model_alias,
        "--model-path",
        args.model_path,
        "--modified-dir",
        args.modified_dir,
        "--neuron-dir",
        args.neuron_dir,
        "--output-dir",
        str(temp_root),
        "--split",
        args.split,
        "--separate-subsets",
        "--max-rounds",
        args.max_rounds,
        "--max-new-tokens",
        args.max_new_tokens,
        "--max-tasks-per-type",
        args.max_tasks_per_type,
        "--seed",
        args.seed,
        "--torch-dtype",
        args.torch_dtype,
        "--device-map",
        args.device_map,
        "--enable-thinking",
        args.enable_thinking,
        "--record-mode",
        args.record_mode,
        "--interventions",
        intervention,
        "--overwrite",
    )
    mgpu.add_list(cmd, "--subsets", [subset])
    mgpu.add_list(cmd, "--task-types", [task_type])
    if args.temperature is not None:
        mgpu.add_kv(cmd, "--temperature", args.temperature)
    if args.top_p is not None:
        mgpu.add_kv(cmd, "--top-p", args.top_p)
    if args.top_k is not None:
        mgpu.add_kv(cmd, "--top-k", args.top_k)
    return mgpu.Job(name=f"stage6-{subset}-{task_type}-{intervention}", cmd=cmd, cuda_device=device)


def copy_unit(args: argparse.Namespace, subset: str, task_type: str, intervention: str) -> None:
    src = unit_root(args, subset, task_type, intervention) / args.model_alias / "single_type_by_subset" / subset / task_type / intervention
    dst = final_root(args) / subset / task_type / intervention
    if unit_complete(args, subset, task_type, intervention) and not args.overwrite:
        print(f"[skip] stage6 unit complete: {dst}", flush=True)
        return
    if not mgpu.complete(src, ("per_task.jsonl", "summary.json")):
        raise FileNotFoundError(f"Missing worker output: {src}")
    mgpu.copytree(src, dst, overwrite=True)
    mgpu.write_json(dst / "runner_meta.json", expected_unit_meta(args, subset, task_type, intervention))


def refresh(args: argparse.Namespace) -> None:
    cmd = mgpu.command(
        "code/05_single_type_causal_validation/run_single_type_causal_validation.py",
        "--model-alias",
        args.model_alias,
        "--model-path",
        args.model_path,
        "--modified-dir",
        args.modified_dir,
        "--neuron-dir",
        args.neuron_dir,
        "--output-dir",
        args.output_dir,
        "--split",
        args.split,
        "--separate-subsets",
        "--max-rounds",
        args.max_rounds,
        "--max-new-tokens",
        args.max_new_tokens,
        "--max-tasks-per-type",
        args.max_tasks_per_type,
        "--seed",
        args.seed,
        "--record-mode",
        args.record_mode,
        "--refresh-existing",
    )
    mgpu.add_list(cmd, "--subsets", args.subsets)
    mgpu.add_list(cmd, "--task-types", args.task_types)
    mgpu.run_refresh(cmd)


def main() -> None:
    args = parse_args()
    mgpu.resolve_model_args(args)
    if stage_complete(args) and not args.overwrite:
        print(f"[skip] stage 6 already complete: {final_root(args)}", flush=True)
        return
    devices = mgpu.parse_devices(args.cuda_devices, args.num_gpus)
    units = [
        (subset, task_type, intervention)
        for subset in args.subsets
        for task_type in args.task_types
        for intervention in interventions_for(task_type)
    ]
    runnable_units: List[tuple[str, str, str]] = []
    for subset, task_type, intervention in units:
        if unit_complete(args, subset, task_type, intervention) and not args.overwrite:
            print(f"[skip] stage6 unit complete before launch: {subset}/{task_type}/{intervention}", flush=True)
            continue
        stale_dir = final_root(args) / subset / task_type / intervention
        if stale_dir.exists():
            print(f"[clean] removing stale or partial stage6 unit: {stale_dir}", flush=True)
            mgpu.remove_if_exists(stale_dir)
        runnable_units.append((subset, task_type, intervention))
    jobs = [
        build_job(args, subset, task_type, intervention, devices[idx % len(devices)])
        for idx, (subset, task_type, intervention) in enumerate(runnable_units)
    ]
    if jobs:
        mgpu.run_jobs(jobs, max_parallel=min(len(jobs), args.num_gpus))
    for subset, task_type, intervention in units:
        copy_unit(args, subset, task_type, intervention)
    refresh(args)
    if not args.keep_workdir:
        mgpu.remove_if_exists(final_root(args) / "_stage6_workers")


if __name__ == "__main__":
    main()
