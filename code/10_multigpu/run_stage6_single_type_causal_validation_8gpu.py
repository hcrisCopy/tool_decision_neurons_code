#!/usr/bin/env python3
"""Single-node multi-GPU runner for stage 6 single-type causal validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common import multigpu_utils as mgpu  # noqa: E402

SUBSETS = ("single_hop", "multi_hop")
TASK_TYPES = ("A", "B", "C")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-alias", default="qwen3-4b-instruct")
    parser.add_argument("--model-path", required=True)
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
    parser.add_argument("--enable-thinking", default="auto", choices=["auto", "true", "false"])
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--record-mode", default="lite", choices=["lite", "full", "off"])
    parser.add_argument("--keep-workdir", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def final_root(args: argparse.Namespace) -> Path:
    return Path(args.output_dir) / args.model_alias / "single_type_by_subset"


def task_complete(args: argparse.Namespace, subset: str, task_type: str) -> bool:
    task_dir = final_root(args) / subset / task_type
    interventions = ["Base", "M-Random", f"M-TDN_{task_type}"]
    return all(mgpu.complete(task_dir / intervention, ("per_task.jsonl", "summary.json")) for intervention in interventions)


def stage_complete(args: argparse.Namespace) -> bool:
    return (
        (final_root(args) / "manifest.json").exists()
        and all(task_complete(args, subset, task_type) for subset in args.subsets for task_type in args.task_types)
    )


def unit_root(args: argparse.Namespace, subset: str, task_type: str) -> Path:
    return final_root(args) / "_stage6_workers" / subset / task_type


def build_job(args: argparse.Namespace, subset: str, task_type: str, device: str) -> mgpu.Job:
    temp_root = unit_root(args, subset, task_type)
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
    return mgpu.Job(name=f"stage6-{subset}-{task_type}", cmd=cmd, cuda_device=device)


def copy_unit(args: argparse.Namespace, subset: str, task_type: str) -> None:
    src = unit_root(args, subset, task_type) / args.model_alias / "single_type_by_subset" / subset / task_type
    dst = final_root(args) / subset / task_type
    if task_complete(args, subset, task_type) and not args.overwrite:
        print(f"[skip] stage6 unit complete: {dst}", flush=True)
        return
    if not src.exists():
        raise FileNotFoundError(f"Missing worker output: {src}")
    mgpu.copytree(src, dst, overwrite=True)


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
    if stage_complete(args) and not args.overwrite:
        print(f"[skip] stage 6 already complete: {final_root(args)}", flush=True)
        return
    devices = mgpu.parse_devices(args.cuda_devices, args.num_gpus)
    jobs: List[mgpu.Job] = []
    units = [(subset, task_type) for subset in args.subsets for task_type in args.task_types]
    for idx, (subset, task_type) in enumerate(units):
        if task_complete(args, subset, task_type) and not args.overwrite:
            print(f"[skip] stage6 unit complete before launch: {subset}/{task_type}", flush=True)
            continue
        jobs.append(build_job(args, subset, task_type, devices[idx % len(devices)]))
    if jobs:
        mgpu.run_jobs(jobs, max_parallel=min(len(jobs), args.num_gpus))
    for subset, task_type in units:
        copy_unit(args, subset, task_type)
    refresh(args)
    if not args.keep_workdir:
        mgpu.remove_if_exists(final_root(args) / "_stage6_workers")


if __name__ == "__main__":
    main()
