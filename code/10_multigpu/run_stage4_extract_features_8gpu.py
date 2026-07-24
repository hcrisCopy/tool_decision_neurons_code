#!/usr/bin/env python3
"""Single-node multi-GPU runner for stage 4 feature extraction."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common import multigpu_utils as mgpu  # noqa: E402

SUBSETS = ("single_hop", "multi_hop")
SPLITS = ("train", "test")
SPLIT_FILES = ("activations.pt", "meta.jsonl", "summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-alias", default="qwen3-4b-instruct")
    parser.add_argument("--model-path", default="", help="Optional. Defaults to configs/models.yaml by --model-alias.")
    parser.add_argument("--models-config", default=None)
    parser.add_argument("--allow-remote-model-download", action="store_true")
    parser.add_argument("--modified-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--subsets", nargs="+", default=list(SUBSETS), choices=SUBSETS)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--cuda-devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--enable-thinking", default="false", choices=["model", "auto", "true", "false"])
    parser.add_argument("--keep-shards", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-shard-index", type=int, default=0)
    parser.add_argument("--worker-num-shards", type=int, default=1)
    parser.add_argument("--worker-output-root", default="")
    return parser.parse_args()


def load_stage4() -> Any:
    path = CODE_ROOT / "03_feature_extraction" / "extract_features.py"
    spec = importlib.util.spec_from_file_location("stage4_extract_features", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def final_split_dir(args: argparse.Namespace, subset: str, split: str) -> Path:
    return Path(args.output_dir) / args.model_alias / subset / split


def shard_split_dir(args: argparse.Namespace, shard_index: int, subset: str, split: str) -> Path:
    root = Path(args.worker_output_root) if args.worker_output_root else Path(args.output_dir) / args.model_alias / "_stage4_shards"
    return root / f"shard_{shard_index:05d}" / subset / split


def stage_complete(args: argparse.Namespace) -> bool:
    return all(mgpu.complete(final_split_dir(args, subset, split), SPLIT_FILES) for subset in args.subsets for split in args.splits)


def run_worker(args: argparse.Namespace) -> None:
    stage4 = load_stage4()
    model_path = Path(args.model_path)
    modified_dir = Path(args.modified_dir)
    enable_thinking = stage4.enable_thinking_value(args.enable_thinking)
    tool_format = stage4.detect_tool_format(str(model_path))
    tokenizer = stage4.AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
    model = stage4.AutoModelForCausalLM.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        torch_dtype=stage4.torch_dtype(args.torch_dtype),
        device_map=args.device_map,
        local_files_only=True,
    ).eval()

    for subset in args.subsets:
        for split in args.splits:
            if mgpu.complete(final_split_dir(args, subset, split), SPLIT_FILES) and not args.overwrite:
                print(f"[skip] final split complete before shard work: {subset}/{split}", flush=True)
                continue
            out_dir = shard_split_dir(args, args.worker_shard_index, subset, split)
            if mgpu.complete(out_dir, SPLIT_FILES) and not args.overwrite:
                print(f"[skip] shard complete: {out_dir}", flush=True)
                continue
            records = stage4.load_records(modified_dir, subset, split)
            selected: List[Tuple[int, Dict[str, Any]]] = [
                (idx, record)
                for idx, record in enumerate(records)
                if idx % args.worker_num_shards == args.worker_shard_index
            ]
            selected_records = [record for _idx, record in selected]
            print(
                f"{subset}/{split}: shard {args.worker_shard_index}/{args.worker_num_shards} "
                f"records={len(selected_records)}",
                flush=True,
            )
            if not selected_records:
                continue
            tensors, meta_rows = stage4.extract_split(model, tokenizer, selected_records, tool_format, enable_thinking)
            for shard_row, (source_idx, _record) in zip(meta_rows, selected):
                shard_row["row_index"] = int(source_idx)
                shard_row["source_row_index"] = int(source_idx)
                shard_row["shard_index"] = int(args.worker_shard_index)
                shard_row["num_shards"] = int(args.worker_num_shards)
            stage4.save_split(out_dir, tensors, meta_rows, overwrite=True)
            print(f"[saved] {out_dir}", flush=True)


def merge_split(args: argparse.Namespace, subset: str, split: str, shard_root: Path) -> Dict[str, Any]:
    import torch

    stage4 = load_stage4()
    out_dir = final_split_dir(args, subset, split)
    if mgpu.complete(out_dir, SPLIT_FILES) and not args.overwrite:
        print(f"[skip] final split complete: {out_dir}", flush=True)
        return {"subset": subset, "split": split, "n": len(mgpu.read_jsonl(out_dir / "meta.jsonl")), "output_dir": str(out_dir), "skipped": True}

    entries: List[Tuple[int, int, Dict[str, Any], Dict[str, Any]]] = []
    records = stage4.load_records(Path(args.modified_dir), subset, split)
    for shard_index in range(args.num_gpus):
        expected = sum(1 for idx in range(len(records)) if idx % args.num_gpus == shard_index)
        shard_dir = shard_root / f"shard_{shard_index:05d}" / subset / split
        if expected == 0 and not shard_dir.exists():
            continue
        if not mgpu.complete(shard_dir, SPLIT_FILES):
            raise FileNotFoundError(f"Missing complete shard output: {shard_dir}")
        tensors = torch.load(shard_dir / "activations.pt", map_location="cpu")
        meta_rows = mgpu.read_jsonl(shard_dir / "meta.jsonl")
        if len(meta_rows) != expected:
            raise ValueError(f"Shard row count mismatch for {shard_dir}: expected {expected}, got {len(meta_rows)}")
        for row_idx, row in enumerate(meta_rows):
            source_idx = int(row.get("source_row_index", row.get("row_index", row_idx)))
            entries.append((source_idx, row_idx, row, tensors))
    entries.sort(key=lambda item: item[0])
    if len(entries) != len(records):
        raise ValueError(f"Merged row count mismatch for {subset}/{split}: expected {len(records)}, got {len(entries)}")

    merged_meta: List[Dict[str, Any]] = []
    tensor_names = list(entries[0][3].keys()) if entries else []
    merged_tensors: Dict[str, Any] = {}
    for new_idx, (source_idx, _row_idx, row, _tensors) in enumerate(entries):
        row = dict(row)
        row["source_row_index"] = int(source_idx)
        row["row_index"] = int(new_idx)
        merged_meta.append(row)
    for name in tensor_names:
        merged_tensors[name] = torch.stack([tensors[name][row_idx] for _source_idx, row_idx, _row, tensors in entries], dim=0)

    stage4.save_split(out_dir, merged_tensors, merged_meta, overwrite=True)
    print(f"[merged] {subset}/{split}: n={len(merged_meta)} -> {out_dir}", flush=True)
    return {"subset": subset, "split": split, "n": len(merged_meta), "output_dir": str(out_dir), "num_shards": args.num_gpus}


def run_parent(args: argparse.Namespace) -> None:
    if stage_complete(args) and not args.overwrite:
        print(f"[skip] stage 4 already complete: {Path(args.output_dir) / args.model_alias}", flush=True)
        return

    devices = mgpu.parse_devices(args.cuda_devices, args.num_gpus)
    shard_root = Path(args.output_dir) / args.model_alias / "_stage4_shards"
    if args.overwrite:
        mgpu.remove_if_exists(shard_root)

    jobs: List[mgpu.Job] = []
    for shard_index, device in enumerate(devices):
        cmd = mgpu.command(
            "code/10_multigpu/run_stage4_extract_features_8gpu.py",
            "--worker",
            "--model-alias",
            args.model_alias,
            "--model-path",
            args.model_path,
            "--modified-dir",
            args.modified_dir,
            "--output-dir",
            args.output_dir,
            "--worker-output-root",
            str(shard_root),
            "--worker-shard-index",
            shard_index,
            "--worker-num-shards",
            args.num_gpus,
            "--torch-dtype",
            args.torch_dtype,
            "--device-map",
            args.device_map,
            "--enable-thinking",
            args.enable_thinking,
        )
        mgpu.add_list(cmd, "--subsets", args.subsets)
        mgpu.add_list(cmd, "--splits", args.splits)
        mgpu.add_flag(cmd, "--overwrite", args.overwrite)
        jobs.append(mgpu.Job(name=f"stage4-shard-{shard_index:05d}", cmd=cmd, cuda_device=device))

    mgpu.run_jobs(jobs, max_parallel=args.num_gpus)
    split_summaries = [merge_split(args, subset, split, shard_root) for subset in args.subsets for split in args.splits]
    mgpu.write_json(
        Path(args.output_dir) / args.model_alias / "manifest.json",
        {
            "stage": "stage4_feature_extraction_multigpu",
            "model_alias": args.model_alias,
            "model_path": args.model_path,
            "modified_dir": args.modified_dir,
            "output_dir": str(Path(args.output_dir) / args.model_alias),
            "num_gpus": args.num_gpus,
            "cuda_devices": devices,
            "prompt_mode": "current",
            "reasoning_mode": "no_reasoning",
            "neuron_definition": "Who Transfers Safety attention projection neurons: Q/K/V projection rows and O projection columns.",
            "saved_tensors": ["z_last", "hidden_last_all_layers", "q_proj_last", "k_proj_last", "v_proj_last", "o_proj_input_last"],
            "splits": split_summaries,
        },
    )
    if not args.keep_shards:
        mgpu.remove_if_exists(shard_root)


def main() -> None:
    args = parse_args()
    mgpu.resolve_model_args(args)
    if args.worker:
        run_worker(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
