#!/usr/bin/env python3
"""Single-node multi-GPU runner for stage 5 single-type neuron probing."""

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
SUBSET_FILES = ("manifest.json",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-alias", default="qwen3-4b-instruct")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--subsets", nargs="+", default=list(SUBSETS), choices=SUBSETS)
    parser.add_argument("--probe-splits", nargs="+", default=["train"], choices=["train", "test"])
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--cuda-devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--top-p", type=float, default=0.03)
    parser.add_argument("--top-preview", type=int, default=50)
    parser.add_argument("--deactivation-batch-size", type=int, default=1)
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--keep-workdir", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def final_root(args: argparse.Namespace) -> Path:
    return Path(args.output_dir) / args.model_alias / "single_type_by_subset"


def subset_complete(args: argparse.Namespace, subset: str) -> bool:
    subset_dir = final_root(args) / subset
    if not mgpu.complete(subset_dir, SUBSET_FILES):
        return False
    return all((subset_dir / task_type / "summary.json").exists() for task_type in ("A", "B", "C"))


def stage_complete(args: argparse.Namespace) -> bool:
    return all(subset_complete(args, subset) for subset in args.subsets) and (final_root(args) / "manifest.json").exists()


def worker_root(args: argparse.Namespace, subset: str) -> Path:
    return final_root(args) / "_stage5_workers" / subset


def build_job(args: argparse.Namespace, subset: str, device: str) -> mgpu.Job:
    temp_root = worker_root(args, subset)
    cmd = mgpu.command(
        "code/04_single_type_neuron_probing/probe_single_type_neurons.py",
        "--model-alias",
        args.model_alias,
        "--model-path",
        args.model_path,
        "--feature-dir",
        args.feature_dir,
        "--output-dir",
        str(temp_root),
        "--separate-subsets",
        "--top-p",
        args.top_p,
        "--top-preview",
        args.top_preview,
        "--deactivation-batch-size",
        args.deactivation_batch_size,
        "--torch-dtype",
        args.torch_dtype,
        "--device-map",
        args.device_map,
    )
    mgpu.add_list(cmd, "--subsets", [subset])
    mgpu.add_list(cmd, "--probe-splits", args.probe_splits)
    mgpu.add_flag(cmd, "--overwrite", args.overwrite)
    return mgpu.Job(name=f"stage5-{subset}", cmd=cmd, cuda_device=device)


def merge_outputs(args: argparse.Namespace) -> None:
    root = final_root(args)
    root.mkdir(parents=True, exist_ok=True)
    children: List[Dict[str, Any]] = []
    for subset in args.subsets:
        if subset_complete(args, subset) and not args.overwrite:
            print(f"[skip] stage5 subset complete: {root / subset}", flush=True)
        else:
            src = worker_root(args, subset) / args.model_alias / "single_type_by_subset" / subset
            dst = root / subset
            if not mgpu.complete(src, SUBSET_FILES):
                raise FileNotFoundError(f"Missing worker output: {src}")
            mgpu.copytree(src, dst, overwrite=True)
        manifest = mgpu.read_json(root / subset / "manifest.json")
        children.append(
            {
                "subset": subset,
                "output_dir": str(root / subset),
                "label_counts": manifest.get("label_counts", {}),
                "candidate_universe": manifest.get("candidate_universe", {}),
                "task_type_summaries": manifest.get("task_type_summaries", []),
            }
        )
    mgpu.write_json(
        root / "manifest.json",
        {
            "stage": "stage5_single_type_neuron_probing_by_subset_multigpu",
            "model_alias": args.model_alias,
            "model_path": args.model_path,
            "feature_dir": args.feature_dir,
            "output_dir": str(root),
            "probe_splits": list(args.probe_splits),
            "subsets": list(args.subsets),
            "neuron_definition": "Who Transfers Safety attention projection neurons: Q/K/V projection rows and O output-projection columns.",
            "importance_definition": "Delta_m(x,N)=||z_m(x)-z_{m,without N}(x)||_2; I_m(N,D)=mean_x Delta_m(x,N).",
            "split_policy": "single_hop and multi_hop are probed independently on the train split.",
            "children": children,
        },
    )


def main() -> None:
    args = parse_args()
    if stage_complete(args) and not args.overwrite:
        print(f"[skip] stage 5 already complete: {final_root(args)}", flush=True)
        return
    devices = mgpu.parse_devices(args.cuda_devices, args.num_gpus)
    jobs: List[mgpu.Job] = []
    for idx, subset in enumerate(args.subsets):
        if subset_complete(args, subset) and not args.overwrite:
            print(f"[skip] stage5 subset complete before launch: {subset}", flush=True)
            continue
        jobs.append(build_job(args, subset, devices[idx % len(devices)]))
    if jobs:
        mgpu.run_jobs(jobs, max_parallel=min(len(jobs), args.num_gpus))
    merge_outputs(args)
    if not args.keep_workdir:
        mgpu.remove_if_exists(final_root(args) / "_stage5_workers")


if __name__ == "__main__":
    main()
