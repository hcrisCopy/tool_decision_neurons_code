#!/usr/bin/env python3
"""Single-node multi-GPU runner for stage 9 CTD neuron training."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-alias", default="qwen3-4b-instruct")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--modified-dir", required=True)
    parser.add_argument("--shared-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--subsets", nargs="+", default=list(SUBSETS), choices=SUBSETS)
    parser.add_argument("--task-types", nargs="+", default=list(TASK_TYPES), choices=TASK_TYPES)
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--cuda-devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--enable-thinking", default="auto", choices=["auto", "true", "false"])
    parser.add_argument("--max-gradient-norm", type=float, default=1.0)
    parser.add_argument("--save-full-selected-param-snapshot", action="store_true")
    parser.add_argument("--keep-workdir", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def final_root(args: argparse.Namespace) -> Path:
    return Path(args.output_dir) / args.model_alias / "neuron_training_by_subset"


def subset_complete(args: argparse.Namespace, subset: str) -> bool:
    subset_dir = final_root(args) / subset
    return mgpu.complete(subset_dir, ("ctd_neuron_delta.pt", "manifest.json", "training_log.csv"))


def stage_complete(args: argparse.Namespace) -> bool:
    return (final_root(args) / "manifest.json").exists() and all(subset_complete(args, subset) for subset in args.subsets)


def worker_root(args: argparse.Namespace, subset: str) -> Path:
    return final_root(args) / "_stage9_workers" / subset


def build_job(args: argparse.Namespace, subset: str, device: str) -> mgpu.Job:
    temp_root = worker_root(args, subset)
    mgpu.remove_if_exists(temp_root)
    cmd = mgpu.command(
        "code/08_training/train_shared_neurons.py",
        "--model-alias",
        args.model_alias,
        "--model-path",
        args.model_path,
        "--modified-dir",
        args.modified_dir,
        "--shared-dir",
        args.shared_dir,
        "--output-dir",
        str(temp_root),
        "--split",
        args.split,
        "--epochs",
        args.epochs,
        "--per-device-train-batch-size",
        args.per_device_train_batch_size,
        "--gradient-accumulation-steps",
        args.gradient_accumulation_steps,
        "--learning-rate",
        args.learning_rate,
        "--warmup-ratio",
        args.warmup_ratio,
        "--max-length",
        args.max_length,
        "--seed",
        args.seed,
        "--torch-dtype",
        args.torch_dtype,
        "--device-map",
        args.device_map,
        "--enable-thinking",
        args.enable_thinking,
        "--max-gradient-norm",
        args.max_gradient_norm,
        "--overwrite",
    )
    mgpu.add_list(cmd, "--subsets", [subset])
    mgpu.add_list(cmd, "--task-types", args.task_types)
    mgpu.add_flag(cmd, "--save-full-selected-param-snapshot", args.save_full_selected_param_snapshot)
    return mgpu.Job(name=f"stage9-{subset}", cmd=cmd, cuda_device=device)


def merge_outputs(args: argparse.Namespace) -> None:
    root = final_root(args)
    root.mkdir(parents=True, exist_ok=True)
    manifests: List[Dict[str, Any]] = []
    for subset in args.subsets:
        if subset_complete(args, subset) and not args.overwrite:
            print(f"[skip] stage9 subset complete: {root / subset}", flush=True)
        else:
            src = worker_root(args, subset) / args.model_alias / "neuron_training_by_subset" / subset
            dst = root / subset
            if not mgpu.complete(src, ("ctd_neuron_delta.pt", "manifest.json", "training_log.csv")):
                raise FileNotFoundError(f"Missing worker output: {src}")
            mgpu.copytree(src, dst, overwrite=True)
        manifests.append(mgpu.read_json(root / subset / "manifest.json"))
    mgpu.write_json(
        root / "manifest.json",
        {
            "stage": "stage9_ctd_neuron_training_multigpu",
            "model_alias": args.model_alias,
            "model_path": args.model_path,
            "subsets": list(args.subsets),
            "output_dir": str(root),
            "num_gpus": args.num_gpus,
            "split": args.split,
            "task_types": list(args.task_types),
            "subset_manifests": manifests,
        },
    )


def main() -> None:
    args = parse_args()
    if stage_complete(args) and not args.overwrite:
        print(f"[skip] stage 9 already complete: {final_root(args)}", flush=True)
        return
    devices = mgpu.parse_devices(args.cuda_devices, args.num_gpus)
    jobs: List[mgpu.Job] = []
    for idx, subset in enumerate(args.subsets):
        if subset_complete(args, subset) and not args.overwrite:
            print(f"[skip] stage9 subset complete before launch: {subset}", flush=True)
            continue
        jobs.append(build_job(args, subset, devices[idx % len(devices)]))
    if jobs:
        mgpu.run_jobs(jobs, max_parallel=min(len(jobs), args.num_gpus))
    merge_outputs(args)
    if not args.keep_workdir:
        mgpu.remove_if_exists(final_root(args) / "_stage9_workers")


if __name__ == "__main__":
    main()
