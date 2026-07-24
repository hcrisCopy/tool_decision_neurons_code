#!/usr/bin/env python3
"""Single-node multi-GPU runner for stage 5 single-type neuron probing."""

from __future__ import annotations

import argparse
import importlib.util
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
SHARD_FILES = ("manifest.json", "score_shard.pt")
EXPECTED_SUBSET_STAGE = "stage5_single_type_neuron_probing_merged_candidate_shards"
EXPECTED_ROOT_STAGE = "stage5_single_type_neuron_probing_by_subset_candidate_sharded_multigpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-alias", default="qwen3-4b-instruct")
    parser.add_argument("--model-path", default="", help="Optional. Defaults to configs/models.yaml by --model-alias.")
    parser.add_argument("--models-config", default=None)
    parser.add_argument("--allow-remote-model-download", action="store_true")
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
    try:
        manifest = mgpu.read_json(subset_dir / "manifest.json")
    except (OSError, ValueError):
        return False
    if manifest.get("stage") != EXPECTED_SUBSET_STAGE:
        print(f"[stale] stage5 subset was not produced by candidate-shard merge: {subset_dir}", flush=True)
        return False
    sharding = manifest.get("candidate_sharding", {})
    if int(sharding.get("num_shards", -1)) != int(args.num_gpus):
        print(f"[stale] stage5 subset shard count changed: {subset_dir}", flush=True)
        return False
    return all((subset_dir / task_type / "summary.json").exists() for task_type in ("A", "B", "C"))


def stage_complete(args: argparse.Namespace) -> bool:
    manifest_path = final_root(args) / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = mgpu.read_json(manifest_path)
    except (OSError, ValueError):
        return False
    return manifest.get("stage") == EXPECTED_ROOT_STAGE and all(subset_complete(args, subset) for subset in args.subsets)


def worker_root(args: argparse.Namespace, subset: str) -> Path:
    return final_root(args) / "_stage5_workers" / subset


def shard_root(args: argparse.Namespace, subset: str, shard_index: int) -> Path:
    return worker_root(args, subset) / f"shard_{shard_index:05d}"


def shard_subset_dir(args: argparse.Namespace, subset: str, shard_index: int) -> Path:
    return shard_root(args, subset, shard_index) / args.model_alias / "single_type_by_subset" / subset


def load_probe_module() -> Any:
    path = CODE_ROOT / "04_single_type_neuron_probing" / "probe_single_type_neurons.py"
    spec = importlib.util.spec_from_file_location("stage5_probe_single_type_neurons", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_job(args: argparse.Namespace, subset: str, shard_index: int, device: str) -> mgpu.Job:
    temp_root = shard_root(args, subset, shard_index)
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
        "--candidate-num-shards",
        args.num_gpus,
        "--candidate-shard-index",
        shard_index,
    )
    mgpu.add_list(cmd, "--subsets", [subset])
    mgpu.add_list(cmd, "--probe-splits", args.probe_splits)
    mgpu.add_flag(cmd, "--overwrite", args.overwrite)
    return mgpu.Job(name=f"stage5-{subset}-shard-{shard_index:05d}", cmd=cmd, cuda_device=device)


def merge_outputs(args: argparse.Namespace) -> None:
    root = final_root(args)
    root.mkdir(parents=True, exist_ok=True)
    probe = load_probe_module()
    children: List[Dict[str, Any]] = []
    for subset in args.subsets:
        if subset_complete(args, subset) and not args.overwrite:
            print(f"[skip] stage5 subset complete: {root / subset}", flush=True)
        else:
            dst = root / subset
            mgpu.remove_if_exists(dst)
            shard_dirs = [shard_subset_dir(args, subset, shard_index) for shard_index in range(args.num_gpus)]
            missing = [
                str(shard_dir)
                for shard_dir in shard_dirs
                if not mgpu.complete(shard_dir, SHARD_FILES)
            ]
            if missing:
                raise FileNotFoundError(f"Missing complete stage5 score shards for {subset}: {missing[:3]}")
            probe.merge_candidate_score_shards(
                args.model_alias,
                args.model_path,
                args.feature_dir,
                dst,
                subset,
                [subset],
                list(args.probe_splits),
                args.top_p,
                args.top_preview,
                shard_dirs,
            )
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
            "stage": EXPECTED_ROOT_STAGE,
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
    mgpu.resolve_model_args(args)
    if stage_complete(args) and not args.overwrite:
        print(f"[skip] stage 5 already complete: {final_root(args)}", flush=True)
        return
    devices = mgpu.parse_devices(args.cuda_devices, args.num_gpus)
    jobs: List[mgpu.Job] = []
    for subset in args.subsets:
        if subset_complete(args, subset) and not args.overwrite:
            print(f"[skip] stage5 subset complete before launch: {subset}", flush=True)
            continue
        if (final_root(args) / subset).exists():
            print(f"[clean] removing stale or partial stage5 subset output: {final_root(args) / subset}", flush=True)
            mgpu.remove_if_exists(final_root(args) / subset)
        mgpu.remove_if_exists(worker_root(args, subset))
        for shard_index, device in enumerate(devices):
            jobs.append(build_job(args, subset, shard_index, device))
    if jobs:
        mgpu.run_jobs(jobs, max_parallel=args.num_gpus)
    merge_outputs(args)
    if not args.keep_workdir:
        mgpu.remove_if_exists(final_root(args) / "_stage5_workers")


if __name__ == "__main__":
    main()
