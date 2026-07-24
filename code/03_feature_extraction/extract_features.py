#!/usr/bin/env python3
"""Stage 4 feature extraction for tool-decision neuron experiments.

It reads the modified dataset, builds the When2Tool
current/no_reasoning prompt, and stores the baseline last-token representation
z_m(x) used by the Who Transfers Safety deactivation score. It also stores
attention projection activations for auditing; Stage 5 uses exact deactivation scores.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
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

SYSTEM_PROMPT_XML = (
    "You can use tools when helpful. "
    "If you call a tool, emit exactly one tool call wrapped in <tool_call>...</tool_call>, "
    "with exactly one JSON object inside: {\"name\": \"tool_name\", \"arguments\": {...}}. "
    "Do not call multiple tools at once. "
    "Tool arguments must strictly match each tool schema; do not invent wrapper fields. "
    "When you are done, provide the final answer in LaTeX boxed format: \\boxed{...}."
)

SYSTEM_PROMPT_NATIVE = (
    "You can use tools when helpful. "
    "Do not call multiple tools at once. "
    "Tool arguments must strictly match each tool schema; do not invent wrapper fields. "
    "When you are done, provide the final answer in LaTeX boxed format: \\boxed{...}."
)

LIST_MANIPULATION_FORMAT_CONTRACT = (
    "ListManipulation format contract:\n"
    "1) There is no set_list tool. For every list-op call, you must provide values=<current list> explicitly.\n"
    "2) For tool arguments, use plain lists [a,b,c] (1D) or [[...],[...]] (2D). Never use objects like {\"values\": [...]}.\n"
    "3) At each step, parse current_list from the latest tool output, then call exactly one operation tool (append/remove/insert/sort/reverse).\n"
    "4) Use one operation per tool call; do not batch multiple operations in one call.\n"
    "5) For final answer, output exactly one plain list literal wrapped in box, e.g. \\boxed{[1, 2, 3]} or \\boxed{[[1, 2], [3, 4]]}; do NOT format it as LaTeX array/matrix.\n"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-alias", default="qwen3-4b-instruct")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--modified-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--subsets", nargs="+", default=list(SUBSETS), choices=SUBSETS)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--enable-thinking", default="false", choices=["auto", "true", "false"])
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


def enable_thinking_value(value: str) -> bool | None:
    if value == "true":
        return True
    if value == "false":
        return False
    return None


def parse_json_field(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, float) and pd.isna(value):
        return default
    if isinstance(value, (list, dict)):
        return value
    text = str(value).strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def detect_tool_format(model_name_or_path: str) -> str:
    return "native" if "llama" in str(model_name_or_path).lower() else "xml"


def system_prompt_for(tool_format: str) -> str:
    return SYSTEM_PROMPT_NATIVE if tool_format == "native" else SYSTEM_PROMPT_XML


def build_current_no_reasoning_user_message(instruction: str) -> str:
    return (
        instruction
        + "\n\nResponse policy (required every turn):\n"
        + "1) You can choose to use a tool or not in this task.\n"
        + "2) Provide final answer in \\boxed{...} if you think the task is complete."
    )


def schema_root() -> Path:
    return REPO_ROOT / "code" / "third_party" / "when2tool_adapter" / "envs"


def build_tools_schema(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    env_name = str(record["env_name"])
    allowed = set(parse_json_field(record.get("tools"), []))
    path = schema_root() / f"{env_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing tool schema: {path}")
    schemas = json.loads(path.read_text(encoding="utf-8"))
    matched = {schema["name"]: schema for schema in schemas}
    missing = sorted(allowed - set(matched))
    if missing:
        raise ValueError(f"{env_name} schema missing tools: {missing}")
    return [{"type": "function", "function": matched[name]} for name in sorted(allowed)]


def messages_for(record: Dict[str, Any], tool_format: str) -> List[Dict[str, str]]:
    messages = [{"role": "system", "content": system_prompt_for(tool_format)}]
    if str(record.get("env_name")) == "ListManipulationEnv":
        messages.append({"role": "system", "content": LIST_MANIPULATION_FORMAT_CONTRACT})
    messages.append({"role": "user", "content": build_current_no_reasoning_user_message(str(record["instruction"]))})
    return messages


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_records(modified_dir: Path, subset: str, split: str) -> List[Dict[str, Any]]:
    path = modified_dir / subset / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing modified dataset split: {path}")
    return list(read_jsonl(path))


def render_prompt(tokenizer: Any, record: Dict[str, Any], tool_format: str, enable_thinking: bool | None) -> str:
    kwargs: Dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    tools = build_tools_schema(record)
    if tools:
        kwargs["tools"] = tools
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    try:
        return tokenizer.apply_chat_template(messages_for(record, tool_format), **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages_for(record, tool_format), **kwargs)


def layer_modules(model: Any) -> List[Any]:
    base = getattr(model, "model", None)
    layers = getattr(base, "layers", None)
    if layers is None:
        raise AttributeError("Cannot find model.model.layers; unsupported architecture for this extractor.")
    return list(layers)


def attach_attention_hooks(model: Any, store: Dict[str, Dict[int, torch.Tensor]]) -> List[Any]:
    hooks = []
    for layer_idx, layer in enumerate(layer_modules(model)):
        attn = getattr(layer, "self_attn")

        def save_output(name: str, idx: int):
            def hook(_module: Any, _inputs: Tuple[Any, ...], output: Any) -> None:
                tensor = output[0] if isinstance(output, tuple) else output
                store[name][idx] = tensor[:, -1, :].detach().cpu().to(torch.float16)

            return hook

        def save_input(name: str, idx: int):
            def hook(_module: Any, inputs: Tuple[Any, ...]) -> None:
                store[name][idx] = inputs[0][:, -1, :].detach().cpu().to(torch.float16)

            return hook

        hooks.append(attn.q_proj.register_forward_hook(save_output("q_proj", layer_idx)))
        hooks.append(attn.k_proj.register_forward_hook(save_output("k_proj", layer_idx)))
        hooks.append(attn.v_proj.register_forward_hook(save_output("v_proj", layer_idx)))
        hooks.append(attn.o_proj.register_forward_pre_hook(save_input("o_proj_input", layer_idx)))
    return hooks


def stack_layer_store(store: Dict[int, torch.Tensor], num_layers: int) -> torch.Tensor:
    missing = [idx for idx in range(num_layers) if idx not in store]
    if missing:
        raise RuntimeError(f"Missing hooked activations for layers: {missing[:5]}")
    return torch.stack([store[idx][0] for idx in range(num_layers)], dim=0)


def extract_split(
    model: Any,
    tokenizer: Any,
    records: List[Dict[str, Any]],
    tool_format: str,
    enable_thinking: bool | None,
) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, Any]]]:
    num_layers = len(layer_modules(model))
    device = model.get_input_embeddings().weight.device
    meta_rows: List[Dict[str, Any]] = []
    hidden_rows: List[torch.Tensor] = []
    q_rows: List[torch.Tensor] = []
    k_rows: List[torch.Tensor] = []
    v_rows: List[torch.Tensor] = []
    o_rows: List[torch.Tensor] = []

    for idx, record in enumerate(tqdm(records, desc="extract", unit="sample"), start=1):
        prompt = render_prompt(tokenizer, record, tool_format, enable_thinking)
        encoded = tokenizer(prompt, return_tensors="pt").to(device)
        store: Dict[str, Dict[int, torch.Tensor]] = defaultdict(dict)
        hooks = attach_attention_hooks(model, store)
        try:
            with torch.no_grad():
                outputs = model(**encoded, output_hidden_states=True, use_cache=False)
        finally:
            for hook in hooks:
                hook.remove()

        hidden_stack = torch.stack([h[0, -1, :].detach().cpu().to(torch.float16) for h in outputs.hidden_states], dim=0)
        hidden_rows.append(hidden_stack)
        q_rows.append(stack_layer_store(store["q_proj"], num_layers))
        k_rows.append(stack_layer_store(store["k_proj"], num_layers))
        v_rows.append(stack_layer_store(store["v_proj"], num_layers))
        o_rows.append(stack_layer_store(store["o_proj_input"], num_layers))
        meta_rows.append(
            {
                "row_index": idx - 1,
                "sample_uid": record.get("sample_uid"),
                "id": record.get("id"),
                "subset": record.get("subset"),
                "split": record.get("split"),
                "env_name": record.get("env_name"),
                "task_type": record.get("task_type"),
                "tool_necessary": record.get("tool_necessary"),
                "no_tool_correct": record.get("no_tool_correct"),
                "prompt_mode": "current",
                "reasoning_mode": "no_reasoning",
                "prompt_tokens": int(encoded["input_ids"].shape[-1]),
                "prompt_text": prompt,
            }
        )

    tensors = {
        "z_last": torch.stack([row[-1] for row in hidden_rows], dim=0),
        "hidden_last_all_layers": torch.stack(hidden_rows, dim=0),
        "q_proj_last": torch.stack(q_rows, dim=0),
        "k_proj_last": torch.stack(k_rows, dim=0),
        "v_proj_last": torch.stack(v_rows, dim=0),
        "o_proj_input_last": torch.stack(o_rows, dim=0),
    }
    return tensors, meta_rows


def save_split(output_dir: Path, tensors: Dict[str, torch.Tensor], meta_rows: List[Dict[str, Any]], overwrite: bool) -> None:
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_dir}. Use --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(tensors, output_dir / "activations.pt")
    write_jsonl(output_dir / "meta.jsonl", meta_rows)
    shapes = {name: list(tensor.shape) for name, tensor in tensors.items()}
    counts = pd.Series([row["task_type"] for row in meta_rows]).value_counts().sort_index().to_dict()
    labels = pd.Series([row["tool_necessary"] for row in meta_rows]).value_counts().sort_index().to_dict()
    write_json(
        output_dir / "summary.json",
        {
            "n": len(meta_rows),
            "tensor_shapes": shapes,
            "task_type_counts": counts,
            "tool_necessary_counts": labels,
        },
    )


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    modified_dir = Path(args.modified_dir)
    output_root = Path(args.output_dir)
    enable_thinking = enable_thinking_value(args.enable_thinking)
    tool_format = detect_tool_format(str(model_path))

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        torch_dtype=torch_dtype(args.torch_dtype),
        device_map=args.device_map,
        local_files_only=True,
    ).eval()

    split_summaries = []
    for subset in args.subsets:
        for split in args.splits:
            records = load_records(modified_dir, subset, split)
            print(f"{subset}/{split}: loaded {len(records)} modified records")
            tensors, meta_rows = extract_split(model, tokenizer, records, tool_format, enable_thinking)
            split_dir = output_root / args.model_alias / subset / split
            save_split(split_dir, tensors, meta_rows, args.overwrite)
            split_summaries.append({"subset": subset, "split": split, "n": len(records), "output_dir": str(split_dir)})
            print(f"{subset}/{split}: saved {split_dir}")

    write_json(
        output_root / args.model_alias / "manifest.json",
        {
            "stage": "stage4_feature_extraction",
            "model_alias": args.model_alias,
            "model_path": str(model_path),
            "modified_dir": str(modified_dir),
            "output_dir": str(output_root / args.model_alias),
            "prompt_mode": "current",
            "reasoning_mode": "no_reasoning",
            "neuron_definition": "Who Transfers Safety attention projection neurons: Q/K/V projection rows and O projection columns.",
            "saved_tensors": [
                "z_last",
                "hidden_last_all_layers",
                "q_proj_last",
                "k_proj_last",
                "v_proj_last",
                "o_proj_input_last",
            ],
            "importance_note": (
                "Stage 5 must compute neuron importance with deactivation: "
                "Delta_m(x,N)=||z_m(x)-z_{m,without N}(x)||_2. Projection activations are only "
                "stored as audit tensors, not as the final importance score."
            ),
            "splits": split_summaries,
        },
    )


if __name__ == "__main__":
    main()
