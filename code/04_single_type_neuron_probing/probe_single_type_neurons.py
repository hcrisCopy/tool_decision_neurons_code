#!/usr/bin/env python3
"""Stage 5 probing for single-type A/B/C tool-decision neurons.

This follows the experimental plan's Who Transfers Safety style
importance score:

    Delta_m(x,N) = || z_m(x) - z_{m,without N}(x) ||_2
    I_m(N,D)    = mean_{x in D} Delta_m(x,N)

Neuron definition:
- Q/K/V neurons are projection-output coordinates, equivalent to rows of W_Q,
  W_K and W_V.
- O neurons are o_proj input coordinates, equivalent to columns of W_O.

The formal pipeline scores the full attention-projection candidate universe.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.io_utils import write_json, write_jsonl  # noqa: E402

SUBSETS = ("single_hop", "multi_hop")
SPLITS = ("train", "test")
TASK_TYPES = ("A", "B", "C")
MATRICES = ("Q", "K", "V", "O")
MATRIX_TO_TENSOR = {
    "Q": "q_proj_last",
    "K": "k_proj_last",
    "V": "v_proj_last",
    "O": "o_proj_input_last",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-alias", default="qwen3-4b-instruct")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--feature-dir", required=True, help="Stage-4 feature root for one model alias.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--subsets", nargs="+", default=list(SUBSETS), choices=SUBSETS)
    parser.add_argument("--probe-splits", nargs="+", default=["train"], choices=SPLITS)
    parser.add_argument("--separate-subsets", action="store_true")
    parser.add_argument("--top-p", type=float, default=0.03)
    parser.add_argument("--top-preview", type=int, default=50)
    parser.add_argument("--deactivation-batch-size", type=int, default=1)
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def torch_dtype(name: str) -> Any:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    return "auto"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_split(feature_dir: Path, subset: str, split: str) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, Any]]]:
    split_dir = feature_dir / subset / split
    act_path = split_dir / "activations.pt"
    meta_path = split_dir / "meta.jsonl"
    if not act_path.exists():
        raise FileNotFoundError(f"Missing activations: {act_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata: {meta_path}")
    tensors = torch.load(act_path, map_location="cpu")
    meta_rows = read_jsonl(meta_path)
    if "z_last" not in tensors:
        raise KeyError(f"{act_path} missing z_last. Rerun stage 4 with the deactivation-aware extractor.")
    n = len(meta_rows)
    if int(tensors["z_last"].shape[0]) != n:
        raise ValueError(f"z_last row count does not match meta rows for {subset}/{split}")
    for name in MATRIX_TO_TENSOR.values():
        if name not in tensors:
            raise KeyError(f"{act_path} missing tensor: {name}")
        if int(tensors[name].shape[0]) != n:
            raise ValueError(f"{name} row count does not match meta rows for {subset}/{split}")
    for row in meta_rows:
        if not row.get("prompt_text"):
            raise KeyError(f"{meta_path} missing prompt_text. Rerun stage 4.")
    return tensors, meta_rows


def load_features(feature_dir: Path, subsets: Iterable[str], splits: Iterable[str]) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, Any]]]:
    all_meta: List[Dict[str, Any]] = []
    buckets: Dict[str, List[torch.Tensor]] = {"z_last": []}
    buckets.update({name: [] for name in MATRIX_TO_TENSOR.values()})
    for subset in subsets:
        for split in splits:
            tensors, meta_rows = load_split(feature_dir, subset, split)
            row_offset = len(all_meta)
            for row_idx, row in enumerate(meta_rows):
                row["merged_row_index"] = row_offset + row_idx
            all_meta.extend(meta_rows)
            for name in buckets:
                buckets[name].append(tensors[name].to(torch.float32))
            print(f"loaded {subset}/{split}: {len(meta_rows)} rows")
    merged = {name: torch.cat(chunks, dim=0) for name, chunks in buckets.items()}
    return merged, all_meta


def layer_modules(model: Any) -> List[Any]:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise AttributeError("Cannot find model.model.layers; unsupported architecture.")
    return list(layers)


def neuron_id(layer: int, matrix: str, index: int) -> str:
    return f"L{layer:02d}.{matrix}.{index:05d}"


def row_key(row: Dict[str, Any]) -> Tuple[int, str, int]:
    return int(row["layer"]), str(row["matrix"]), int(row["index"])


def candidate_key(candidate: Tuple[int, str, int]) -> str:
    layer, matrix, index = candidate
    return f"{layer}:{matrix}:{index}"


def build_candidate_universe(
    tensors: Dict[str, torch.Tensor],
) -> Tuple[List[Tuple[int, str, int]], Dict[str, Any]]:
    dims = {
        matrix: (int(tensors[tensor_name].shape[1]), int(tensors[tensor_name].shape[2]))
        for matrix, tensor_name in MATRIX_TO_TENSOR.items()
    }
    num_layers = dims["Q"][0]
    candidates: List[Tuple[int, str, int]] = []
    mode = "full"
    for layer_idx in range(num_layers):
        for matrix in MATRICES:
            dim = dims[matrix][1]
            indices = list(range(dim))
            candidates.extend((layer_idx, matrix, int(index)) for index in indices)
    by_layer = Counter(layer for layer, _matrix, _index in candidates)
    by_matrix = Counter(matrix for _layer, matrix, _index in candidates)
    return candidates, {
        "mode": mode,
        "num_candidates": len(candidates),
        "num_layers": num_layers,
        "matrix_dims": {matrix: list(shape) for matrix, shape in dims.items()},
        "candidate_counts_by_layer": {str(layer): count for layer, count in sorted(by_layer.items())},
        "candidate_counts_by_matrix": {matrix: by_matrix[matrix] for matrix in MATRICES},
    }


def selected_mask(rows: List[Dict[str, Any]], dims: Dict[str, Tuple[int, int]]) -> Dict[str, torch.Tensor]:
    masks = {matrix: torch.zeros(shape, dtype=torch.bool) for matrix, shape in dims.items()}
    for row in rows:
        masks[str(row["matrix"])][int(row["layer"]), int(row["index"])] = True
    return masks


def attach_batch_deactivation_hooks(
    model: Any,
    candidate_chunk: List[Tuple[int, str, int]],
) -> List[Any]:
    hooks = []
    grouped: Dict[Tuple[int, str], List[Tuple[int, int]]] = defaultdict(list)
    for batch_idx, (layer_idx, matrix, index) in enumerate(candidate_chunk):
        grouped[(layer_idx, matrix)].append((batch_idx, index))

    for layer_idx, layer in enumerate(layer_modules(model)):
        attn = layer.self_attn

        def save_output(matrix: str, idx: int):
            edits = grouped.get((idx, matrix), [])

            def hook(_module: Any, _inputs: Tuple[Any, ...], output: Any) -> Any:
                if not edits:
                    return output
                tensor = output[0] if isinstance(output, tuple) else output
                edited = tensor.clone()
                for batch_idx, neuron_idx in edits:
                    edited[batch_idx, :, neuron_idx] = 0
                if isinstance(output, tuple):
                    return (edited,) + tuple(output[1:])
                return edited

            return hook

        def save_input(idx: int):
            edits = grouped.get((idx, "O"), [])

            def hook(_module: Any, inputs: Tuple[Any, ...]) -> Tuple[Any, ...]:
                if not edits:
                    return inputs
                edited = inputs[0].clone()
                for batch_idx, neuron_idx in edits:
                    edited[batch_idx, :, neuron_idx] = 0
                return (edited,) + tuple(inputs[1:])

            return hook

        hooks.append(attn.q_proj.register_forward_hook(save_output("Q", layer_idx)))
        hooks.append(attn.k_proj.register_forward_hook(save_output("K", layer_idx)))
        hooks.append(attn.v_proj.register_forward_hook(save_output("V", layer_idx)))
        hooks.append(attn.o_proj.register_forward_pre_hook(save_input(layer_idx)))
    return hooks


def masked_z_for_chunk(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    candidate_chunk: List[Tuple[int, str, int]],
) -> torch.Tensor:
    device = model.get_input_embeddings().weight.device
    encoded = tokenizer(prompt_text, return_tensors="pt")
    batch_size = len(candidate_chunk)
    batch = {
        name: tensor.repeat(batch_size, 1).to(device)
        for name, tensor in encoded.items()
    }
    base_model = getattr(model, "model", None)
    if base_model is None:
        raise AttributeError("Cannot find model.model for representation extraction.")
    hooks = attach_batch_deactivation_hooks(model, candidate_chunk)
    try:
        with torch.no_grad():
            outputs = base_model(**batch, use_cache=False)
            z = outputs.last_hidden_state[:, -1, :].detach().cpu().to(torch.float32)
    finally:
        for hook in hooks:
            hook.remove()
    return z


def compute_deactivation_scores(
    model: Any,
    tokenizer: Any,
    tensors: Dict[str, torch.Tensor],
    meta_rows: List[Dict[str, Any]],
    candidates: List[Tuple[int, str, int]],
    batch_size: int,
) -> Tuple[Dict[Tuple[str, int], Dict[Tuple[int, str, int], float]], Dict[Tuple[str, int], int]]:
    sums: Dict[Tuple[str, int], torch.Tensor] = {
        (task_type, label): torch.zeros(len(candidates), dtype=torch.float64)
        for task_type in TASK_TYPES
        for label in (0, 1)
    }
    counts: Dict[Tuple[str, int], int] = {(task_type, label): 0 for task_type in TASK_TYPES for label in (0, 1)}

    z_all = tensors["z_last"].to(torch.float32)
    for row_idx, row in enumerate(tqdm(meta_rows, desc="deactivation rows", unit="sample"), start=1):
        task_type = str(row["task_type"])
        label = int(row["tool_necessary"])
        group = (task_type, label)
        baseline_z = z_all[int(row["merged_row_index"])]
        counts[group] += 1
        prompt_text = str(row["prompt_text"])
        for start in tqdm(range(0, len(candidates), batch_size), desc=f"neurons {row_idx}/{len(meta_rows)}", leave=False, unit="batch"):
            chunk = candidates[start : start + batch_size]
            masked_z = masked_z_for_chunk(model, tokenizer, prompt_text, chunk)
            diffs = torch.linalg.vector_norm(masked_z - baseline_z.unsqueeze(0), ord=2, dim=1).to(torch.float64)
            sums[group][start : start + len(chunk)] += diffs

    score_maps: Dict[Tuple[str, int], Dict[Tuple[int, str, int], float]] = {}
    for group, values in sums.items():
        denom = max(1, counts[group])
        means = values / denom
        score_maps[group] = {
            candidate: float(means[idx].item())
            for idx, candidate in enumerate(candidates)
        }
    return score_maps, counts


def top_p_for_layer(
    score_map: Dict[Tuple[int, str, int], float],
    candidates_by_layer: Dict[int, List[Tuple[int, str, int]]],
    layer_idx: int,
    top_p: float,
) -> List[Dict[str, Any]]:
    layer_candidates = candidates_by_layer[layer_idx]
    top_n = max(1, math.ceil(len(layer_candidates) * top_p))
    ranked = sorted(layer_candidates, key=lambda item: score_map[item], reverse=True)[:top_n]
    rows: List[Dict[str, Any]] = []
    for rank, (layer, matrix, index) in enumerate(ranked, start=1):
        rows.append(
            {
                "layer": int(layer),
                "matrix": matrix,
                "index": int(index),
                "score": float(score_map[(layer, matrix, index)]),
                "rank_in_layer": rank,
                "neuron_id": neuron_id(layer, matrix, index),
            }
        )
    return rows


def rows_from_keys(
    keys: Iterable[Tuple[int, str, int]],
    score_lookup: Dict[Tuple[int, str, int], float],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for layer, matrix, index in sorted(keys):
        rows.append(
            {
                "layer": int(layer),
                "matrix": matrix,
                "index": int(index),
                "score_label_1": float(score_lookup[(layer, matrix, index)]),
                "neuron_id": neuron_id(layer, matrix, index),
            }
        )
    return rows


def save_type_outputs(
    output_dir: Path,
    task_type: str,
    scores_1: Dict[Tuple[int, str, int], float],
    scores_0: Dict[Tuple[int, str, int], float],
    top_1: List[Dict[str, Any]],
    top_0: List[Dict[str, Any]],
    tdn_rows: List[Dict[str, Any]],
    dims: Dict[str, Tuple[int, int]],
    summary: Dict[str, Any],
) -> None:
    type_dir = output_dir / task_type
    type_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(type_dir / "S1_top_neurons.jsonl", top_1)
    write_jsonl(type_dir / "S0_top_neurons.jsonl", top_0)
    write_jsonl(type_dir / "TDN_neurons.jsonl", tdn_rows)
    write_json(type_dir / "summary.json", summary)

    payload: Dict[str, Any] = {
        "scores_label_1": [
            {"layer": layer, "matrix": matrix, "index": index, "score": score}
            for (layer, matrix, index), score in sorted(scores_1.items())
        ],
        "scores_label_0": [
            {"layer": layer, "matrix": matrix, "index": index, "score": score}
            for (layer, matrix, index), score in sorted(scores_0.items())
        ],
        "selected_S1": selected_mask(top_1, dims),
        "selected_S0": selected_mask(top_0, dims),
        "selected_TDN": selected_mask(tdn_rows, dims),
    }
    torch.save(payload, type_dir / "scores_and_masks.pt")


def probe_scope(
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    feature_dir: Path,
    output_root: Path,
    subsets: List[str],
    subset_scope: str,
) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    tensors, meta_rows = load_features(feature_dir, subsets, args.probe_splits)
    sample_counts = Counter((str(row.get("task_type")), int(row.get("tool_necessary"))) for row in meta_rows)
    print(f"{subset_scope}: label counts: {dict(sorted(sample_counts.items()))}")

    candidates, candidate_meta = build_candidate_universe(tensors)
    candidates_by_layer: Dict[int, List[Tuple[int, str, int]]] = defaultdict(list)
    for candidate in candidates:
        candidates_by_layer[candidate[0]].append(candidate)
    dims = {
        matrix: (int(tensors[tensor_name].shape[1]), int(tensors[tensor_name].shape[2]))
        for matrix, tensor_name in MATRIX_TO_TENSOR.items()
    }
    print(
        f"{subset_scope}: candidate_mode={candidate_meta['mode']}, "
        f"candidates={len(candidates)}, batch_size={args.deactivation_batch_size}"
    )

    score_maps, label_counts = compute_deactivation_scores(
        model,
        tokenizer,
        tensors,
        meta_rows,
        candidates,
        args.deactivation_batch_size,
    )

    manifest_summaries: List[Dict[str, Any]] = []
    for task_type in TASK_TYPES:
        scores_1 = score_maps[(task_type, 1)]
        scores_0 = score_maps[(task_type, 0)]
        top_1: List[Dict[str, Any]] = []
        top_0: List[Dict[str, Any]] = []
        for layer_idx in sorted(candidates_by_layer):
            top_1.extend(top_p_for_layer(scores_1, candidates_by_layer, layer_idx, args.top_p))
            top_0.extend(top_p_for_layer(scores_0, candidates_by_layer, layer_idx, args.top_p))

        top_1_keys = {row_key(row) for row in top_1}
        top_0_keys = {row_key(row) for row in top_0}
        tdn_keys = top_1_keys - top_0_keys
        score_lookup = {row_key(row): float(row["score"]) for row in top_1}
        tdn_rows = rows_from_keys(tdn_keys, score_lookup) if tdn_keys else []

        preview_1 = sorted(top_1, key=lambda row: row["score"], reverse=True)[: args.top_preview]
        preview_0 = sorted(top_0, key=lambda row: row["score"], reverse=True)[: args.top_preview]
        preview_tdn = sorted(tdn_rows, key=lambda row: row["score_label_1"], reverse=True)[: args.top_preview]
        status = "ok" if label_counts[(task_type, 1)] > 0 and label_counts[(task_type, 0)] > 0 else "skipped_insufficient_labels"
        summary = {
            "task_type": task_type,
            "status": status,
            "n_label_1": int(label_counts[(task_type, 1)]),
            "n_label_0": int(label_counts[(task_type, 0)]),
            "top_p": args.top_p,
            "num_layers": candidate_meta["num_layers"],
            "candidate_mode": candidate_meta["mode"],
            "candidate_count": len(candidates),
            "S1_size": len(top_1),
            "S0_size": len(top_0),
            "TDN_size": len(tdn_rows),
            "preview_S1": preview_1,
            "preview_S0": preview_0,
            "preview_TDN": preview_tdn,
        }
        save_type_outputs(output_root, task_type, scores_1, scores_0, top_1, top_0, tdn_rows, dims, summary)
        manifest_summaries.append({key: value for key, value in summary.items() if not key.startswith("preview_")})
        print(
            f"{subset_scope}/{task_type}: status={status}, "
            f"n1={label_counts[(task_type, 1)]}, n0={label_counts[(task_type, 0)]}, "
            f"S1={len(top_1)}, S0={len(top_0)}, TDN={len(tdn_rows)}"
        )

    manifest = {
        "stage": "stage5_single_type_neuron_probing",
        "model_alias": args.model_alias,
        "model_path": args.model_path,
        "feature_dir": str(feature_dir),
        "output_dir": str(output_root),
        "probe_splits": list(args.probe_splits),
        "subsets": list(subsets),
        "subset_scope": subset_scope,
        "separated_by_hop": args.separate_subsets,
        "neuron_definition": "Who Transfers Safety attention projection neurons: Q/K/V projection rows and O output-projection columns.",
        "importance_definition": "Delta_m(x,N)=||z_m(x)-z_{m,without N}(x)||_2; I_m(N,D)=mean_x Delta_m(x,N).",
        "mask_implementation": "Q/K/V zero projection-output coordinates; O zero o_proj input coordinates, equivalent to masking W_O columns.",
        "contrastive_definition": "TDN_{m,c,l}=TopP(I(N,D^1_{m,c})) minus TopP(I(N,D^0_{m,c}))",
        "top_p": args.top_p,
        "matrix_dims": {matrix: list(shape) for matrix, shape in dims.items()},
        "candidate_universe": candidate_meta,
        "label_counts": {f"{task_type}:{label}": count for (task_type, label), count in sorted(sample_counts.items())},
        "task_type_summaries": manifest_summaries,
    }
    write_json(output_root / "manifest.json", manifest)
    print(f"saved stage-5 outputs: {output_root}")
    return manifest


def main() -> None:
    args = parse_args()
    if not (0 < args.top_p <= 1):
        raise ValueError("--top-p must be in (0, 1].")
    if args.deactivation_batch_size <= 0:
        raise ValueError("--deactivation-batch-size must be positive.")

    feature_dir = Path(args.feature_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype(args.torch_dtype),
        device_map=args.device_map,
        local_files_only=True,
    ).eval()

    if args.separate_subsets:
        base_root = Path(args.output_dir) / args.model_alias / "single_type_by_subset"
        if base_root.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {base_root}. Use --overwrite.")
        base_root.mkdir(parents=True, exist_ok=True)
        child_manifests = []
        for subset in args.subsets:
            child_manifest = probe_scope(args, model, tokenizer, feature_dir, base_root / subset, [subset], subset)
            child_manifests.append(
                {
                    "subset": subset,
                    "output_dir": str(base_root / subset),
                    "label_counts": child_manifest["label_counts"],
                    "candidate_universe": child_manifest["candidate_universe"],
                    "task_type_summaries": child_manifest["task_type_summaries"],
                }
            )
        write_json(
            base_root / "manifest.json",
            {
                "stage": "stage5_single_type_neuron_probing_by_subset",
                "model_alias": args.model_alias,
                "model_path": args.model_path,
                "feature_dir": str(feature_dir),
                "output_dir": str(base_root),
                "probe_splits": list(args.probe_splits),
                "subsets": list(args.subsets),
                "neuron_definition": "Who Transfers Safety attention projection neurons: Q/K/V projection rows and O output-projection columns.",
                "importance_definition": "Delta_m(x,N)=||z_m(x)-z_{m,without N}(x)||_2; I_m(N,D)=mean_x Delta_m(x,N).",
                "split_policy": "single_hop and multi_hop are probed independently on the train split.",
                "children": child_manifests,
            },
        )
        print(f"saved separated stage-5 outputs: {base_root}")
        return

    output_root = Path(args.output_dir) / args.model_alias / "single_type"
    if output_root.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_root}. Use --overwrite.")
    probe_scope(args, model, tokenizer, feature_dir, output_root, list(args.subsets), "combined")


if __name__ == "__main__":
    main()
