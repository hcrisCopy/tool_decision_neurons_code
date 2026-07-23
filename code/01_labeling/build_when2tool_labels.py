#!/usr/bin/env python3
"""Generate model-specific When2Tool tool_necessary labels.

This stage follows the original When2Tool label definition:

    tool_necessary = 0 if the model answers correctly in hard_no_tool mode else 1

The input dataset is the raw When2Tool parquet dataset, not the modified
A/B/C-mapped dataset. A/B/C task_type is added only as output metadata so later
stages can group labels without re-reading the mapping.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.env_type_mapping import get_task_type_meta  # noqa: E402
from common.io_utils import write_json, write_jsonl  # noqa: E402
from common.model_registry import list_model_aliases, resolve_model_spec  # noqa: E402


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

SUBSETS = ("single_hop", "multi_hop")
SPLITS = ("train", "test")
RAW_COLUMNS = (
    "id",
    "difficulty",
    "multi_step",
    "instruction",
    "env_name",
    "tools",
    "parameters",
    "answer",
    "steps",
    "tags",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-alias", default="qwen3-1.7b", help="Model alias from configs/models.yaml, or 'all'.")
    parser.add_argument("--model-path", default="", help="Override model path/repo_id for a single model run.")
    parser.add_argument("--models-config", default=None)
    parser.add_argument("--data-root", default="../tool_decision_neurons_data")
    parser.add_argument("--raw-dir", default="", help="Default: <data-root>/datasets/raw_when2tool")
    parser.add_argument("--output-root", default="", help="Default: <data-root>/labels")
    parser.add_argument("--subsets", nargs="+", default=list(SUBSETS), choices=SUBSETS)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    parser.add_argument("--backend", default="hf", choices=["hf", "vllm"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=0, help="Demo/debug only. 0 means all samples.")
    parser.add_argument("--max-rounds", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--vllm-dtype", default="bfloat16")
    parser.add_argument("--enable-thinking", default="false", choices=["auto", "true", "false"])
    parser.add_argument("--prefer-local", action="store_true", default=True)
    parser.add_argument("--no-prefer-local", action="store_false", dest="prefer_local")
    parser.add_argument("--check-data-only", action="store_true", help="Validate raw parquet loading and tool schemas without loading a model.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_model_key(name: str) -> str:
    return str(name or "").strip().lower().replace("_", "-")


def detect_tool_format(model_name_or_path: str) -> str:
    key = normalize_model_key(model_name_or_path)
    return "native" if "llama" in key else "xml"


def system_prompt_for(tool_format: str) -> str:
    return SYSTEM_PROMPT_NATIVE if tool_format == "native" else SYSTEM_PROMPT_XML


def parse_json_field(value: Any, default: Any) -> Any:
    if value is None:
        return deepcopy(default)
    if isinstance(value, float) and math.isnan(value):
        return deepcopy(default)
    if isinstance(value, (list, dict)):
        return deepcopy(value)
    text = str(value).strip()
    if not text:
        return deepcopy(default)
    return json.loads(text)


def load_raw_split(raw_dir: Path, subset: str, split: str, max_samples: int = 0) -> List[Dict[str, Any]]:
    split_dir = raw_dir / subset
    files = sorted(split_dir.glob(f"{split}-*.parquet")) or sorted(split_dir.glob(f"{split}*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet file found for {subset}/{split}: {split_dir}")
    df = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    missing = [col for col in RAW_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required raw columns for {subset}/{split}: {missing}")
    if max_samples > 0:
        df = df.head(max_samples)

    tasks = []
    for row in df.to_dict(orient="records"):
        tools = parse_json_field(row["tools"], [])
        parameters = parse_json_field(row["parameters"], {})
        steps = parse_json_field(row["steps"], [])
        tags = parse_json_field(row["tags"], [])
        expected = {"answer": row["answer"]}
        if steps:
            expected["steps"] = steps
        env_name = str(row["env_name"])
        task_id = row["id"].item() if hasattr(row["id"], "item") else row["id"]
        task = {
            "id": task_id,
            "source_dataset": "When2Tool",
            "subset": subset,
            "split": split,
            "sample_uid": f"{subset}:{split}:{task_id}",
            "difficulty": str(row["difficulty"]),
            "multi_step": bool(row["multi_step"]),
            "instruction": str(row["instruction"]),
            "environments": [{"name": env_name, "tools": tools, "parameters": parameters}],
            "expected": expected,
            "tags": tags,
        }
        task.update(get_task_type_meta(env_name))
        tasks.append(task)
    return tasks


def schema_root() -> Path:
    return REPO_ROOT / "code" / "third_party" / "when2tool_adapter" / "env_schemas"


def load_env_schemas(env_name: str) -> List[Dict[str, Any]]:
    path = schema_root() / f"{env_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing When2Tool env schema: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_tools_schema(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    tools = []
    for env_cfg in task.get("environments", []):
        env_name = env_cfg["name"]
        allowed = set(env_cfg.get("tools") or [])
        schemas = load_env_schemas(env_name)
        matched = {schema["name"]: schema for schema in schemas}
        missing = sorted(allowed - set(matched))
        if missing:
            raise ValueError(f"{env_name} schema missing tools: {missing}")
        for tool_name in sorted(allowed):
            tools.append({"type": "function", "function": deepcopy(matched[tool_name])})
    return tools


def build_user_message(task_instruction: str) -> str:
    return (
        task_instruction
        + "\n\nResponse policy (required every turn):\n"
        + "1) Do not use any tools in this task.\n"
        + "2) Provide final answer in \\boxed{...} if you think the task is complete."
    )


def clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def extract_boxed(text: Any) -> str:
    t = clean(text)
    marker = "\\boxed{"
    idx = t.find(marker)
    if idx < 0:
        return ""
    i = idx + len(marker)
    depth = 1
    out = []
    while i < len(t):
        ch = t[i]
        if ch == "{":
            depth += 1
            out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
            out.append(ch)
        else:
            out.append(ch)
        i += 1
    return "".join(out).strip() if out else t


def normalize_scalar(value: Any) -> str:
    text = re.sub(r"\s+", " ", clean(value)).strip()
    if len(text) >= 2 and ((text[0] == "'" and text[-1] == "'") or (text[0] == '"' and text[-1] == '"')):
        text = text[1:-1].strip()
    text = re.sub(r"\\text\s*\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\mathrm\s*\{([^}]*)\}", r"\1", text)
    text = text.replace("{", "").replace("}", "")
    text = text.replace("\\", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def normalize_structured(value: Any) -> Any:
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return [normalize_structured(v) for v in value]
    if isinstance(value, dict):
        return {k: normalize_structured(v) for k, v in value.items()}
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"[-+]?\d+", text):
            try:
                return int(text)
            except Exception:
                return text
        if re.fullmatch(r"[-+]?\d*\.\d+", text):
            try:
                return float(text)
            except Exception:
                return text
        return text
    return value


def parse_structured(text: Any) -> Any:
    value = clean(text)
    if not value:
        return None
    try:
        return normalize_structured(ast.literal_eval(value))
    except Exception:
        return None


def compare_values(prediction: Any, gold: Any) -> bool:
    pred_struct = parse_structured(prediction)
    gold_struct = parse_structured(gold)
    if pred_struct is not None and gold_struct is not None:
        return pred_struct == gold_struct
    pred = normalize_scalar(prediction)
    target = normalize_scalar(gold)
    return bool(target) and pred == target


def has_boxed_answer(text: Any) -> bool:
    return "\\boxed{" in str(text or "")


def _clean_reasoning_text(text: Any) -> str:
    value = re.sub(r"\\boxed\s*\{[\s\S]*?\}", " ", str(text or ""))
    return re.sub(r"\s+", " ", value).strip()


def _has_nontrivial_reasoning(text: Any) -> bool:
    value = _clean_reasoning_text(text)
    return len(value) >= 12 and re.search(r"[A-Za-z]", value) is not None


def has_reasoning_for_direct_answer(raw_text: Any) -> bool:
    text = str(raw_text or "")
    idx = text.find("\\boxed{")
    if idx >= 0:
        return _has_nontrivial_reasoning(text[:idx])
    return _has_nontrivial_reasoning(text)


def _try_json_block(text: str) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(text)
    except Exception:
        try:
            data = json.loads(text.replace("'", '"'))
        except Exception:
            try:
                data = ast.literal_eval(text)
                json.dumps(data)
            except Exception:
                return None
    if not isinstance(data, dict) or "name" not in data:
        return None
    if "arguments" not in data:
        data["arguments"] = data.get("parameters", {})
    if isinstance(data["arguments"], str):
        try:
            data["arguments"] = json.loads(data["arguments"])
        except Exception:
            data["arguments"] = {}
    if not isinstance(data["arguments"], dict):
        data["arguments"] = {}
    return data


def parse_tool_call_from_text(text: Any) -> Optional[Dict[str, Any]]:
    value = str(text or "").strip().replace("```json", "").replace("```", "").strip()
    if "<tool_call>" in value:
        body = value.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0].strip()
        parsed = _try_json_block(body)
        if parsed is not None:
            return parsed
    parsed = _try_json_block(value)
    if parsed is not None:
        return parsed
    for match in re.finditer(r"\{[\s\S]*?\}", value):
        parsed = _try_json_block(match.group(0))
        if parsed is not None:
            return parsed
    return None


def normalize_generation_output(raw_text: str) -> Dict[str, Any]:
    parsed = parse_tool_call_from_text(raw_text)
    if parsed is not None:
        return {
            "type": "tool",
            "tool_name": parsed["name"],
            "arguments": parsed.get("arguments", {}),
            "raw_text": raw_text,
        }
    return {"type": "content", "content": raw_text, "raw_text": raw_text}


def enable_thinking_value(value: str) -> Optional[bool]:
    if value == "true":
        return True
    if value == "false":
        return False
    return None


def torch_dtype_from_name(name: str) -> Any:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    return "auto"


class HFGenerator:
    def __init__(self, model_path: str, args: argparse.Namespace):
        self.max_new_tokens = args.max_new_tokens
        self.enable_thinking = enable_thinking_value(args.enable_thinking)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype_from_name(args.torch_dtype),
            device_map=args.device_map,
        ).eval()
        self.generation_config = GenerationConfig.from_pretrained(model_path)
        if args.temperature is not None:
            self.generation_config.temperature = float(args.temperature)
        if args.top_p is not None:
            self.generation_config.top_p = float(args.top_p)
        if args.top_k is not None:
            self.generation_config.top_k = int(args.top_k)

    def render_prompt(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]]) -> str:
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if tools:
            kwargs["tools"] = tools
        if self.enable_thinking is not None:
            kwargs["enable_thinking"] = self.enable_thinking
        try:
            return self.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def generate_batch(self, messages_batch: List[List[Dict[str, str]]], tools_batch: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        outputs = []
        embed_device = self.model.get_input_embeddings().weight.device
        for messages, tools in zip(messages_batch, tools_batch):
            prompt_text = self.render_prompt(messages, tools)
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(embed_device)
            generated = self.model.generate(
                **inputs,
                generation_config=deepcopy(self.generation_config),
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            raw_text = self.tokenizer.decode(
                generated[0][len(inputs["input_ids"][0]) :],
                skip_special_tokens=True,
            )
            out = normalize_generation_output(raw_text)
            out["prompt_text"] = prompt_text
            out["finish_reason"] = "length_or_eos"
            outputs.append(out)
        return outputs


class VLLMGenerator:
    def __init__(self, model_path: str, args: argparse.Namespace):
        from vllm import LLM, SamplingParams

        self.SamplingParams = SamplingParams
        self.max_new_tokens = args.max_new_tokens
        self.enable_thinking = enable_thinking_value(args.enable_thinking)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.generation_config = GenerationConfig.from_pretrained(model_path)
        if args.temperature is not None:
            self.generation_config.temperature = float(args.temperature)
        if args.top_p is not None:
            self.generation_config.top_p = float(args.top_p)
        if args.top_k is not None:
            self.generation_config.top_k = int(args.top_k)
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            dtype=args.vllm_dtype,
        )

    def sampling_params(self) -> Any:
        kwargs = {"n": 1, "max_tokens": self.max_new_tokens}
        if getattr(self.generation_config, "temperature", None) is not None:
            kwargs["temperature"] = float(self.generation_config.temperature)
        if getattr(self.generation_config, "top_p", None) is not None:
            kwargs["top_p"] = float(self.generation_config.top_p)
        if getattr(self.generation_config, "top_k", None) is not None:
            kwargs["top_k"] = int(self.generation_config.top_k)
        if getattr(self.generation_config, "repetition_penalty", None) is not None:
            kwargs["repetition_penalty"] = float(self.generation_config.repetition_penalty)
        return self.SamplingParams(**kwargs)

    def render_prompt(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]]) -> str:
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if tools:
            kwargs["tools"] = tools
        if self.enable_thinking is not None:
            kwargs["enable_thinking"] = self.enable_thinking
        try:
            return self.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def generate_batch(self, messages_batch: List[List[Dict[str, str]]], tools_batch: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        prompts = [self.render_prompt(messages, tools) for messages, tools in zip(messages_batch, tools_batch)]
        raw_outputs = self.llm.generate(prompts=prompts, sampling_params=self.sampling_params())
        outputs = []
        for prompt_text, raw in zip(prompts, raw_outputs):
            text = raw.outputs[0].text if raw.outputs else ""
            out = normalize_generation_output(text)
            out["prompt_text"] = prompt_text
            out["finish_reason"] = raw.outputs[0].finish_reason if raw.outputs else "empty"
            outputs.append(out)
        return outputs


def build_generator(model_path: str, args: argparse.Namespace) -> Any:
    if args.backend == "hf":
        return HFGenerator(model_path, args)
    return VLLMGenerator(model_path, args)


def initial_state(task: Dict[str, Any], system_prompt: str, tool_format: str) -> Dict[str, Any]:
    return {
        "task": deepcopy(task),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_user_message(task["instruction"])},
        ],
        "tools": build_tools_schema(task),
        "rounds": 0,
        "done": False,
        "trace": [],
        "final_response": "",
        "tool_calls": 0,
        "tool_format": tool_format,
    }


def step_state(state: Dict[str, Any], out: Dict[str, Any]) -> None:
    state["rounds"] += 1
    raw_text = out.get("raw_text", out.get("content", ""))
    state["messages"].append({"role": "assistant", "content": raw_text})
    trace_item = {
        "round": state["rounds"],
        "prompt_text": out.get("prompt_text", ""),
        "model_raw_output": raw_text,
        "model_finish_reason": out.get("finish_reason", "unknown"),
        "parsed_output": {
            "type": out.get("type", "unknown"),
            "tool_name": out.get("tool_name"),
            "arguments": deepcopy(out.get("arguments", {})) if out.get("type") == "tool" else None,
            "content": out.get("content", "") if out.get("type") == "content" else None,
        },
    }

    if out["type"] == "tool":
        state["messages"].append(
            {
                "role": "user",
                "content": (
                    "Tool use is not available. "
                    "Solve the problem directly without tools and provide your final answer in \\boxed{...}."
                ),
            }
        )
        trace_item["tool_result"] = {
            "success": False,
            "message": "Tool call rejected: hard_no_tool mode.",
        }
        trace_item["state_done_after_step"] = False
        state["trace"].append(trace_item)
        return

    if has_boxed_answer(raw_text):
        if has_reasoning_for_direct_answer(raw_text):
            state["messages"].append(
                {
                    "role": "user",
                    "content": (
                        "Final answer rejected: reasoning is not allowed in no_reasoning mode. "
                        "Retry with final answer in \\boxed{...} only."
                    ),
                }
            )
            trace_item["tool_result"] = None
            trace_item["state_done_after_step"] = False
            state["trace"].append(trace_item)
            return
        state["final_response"] = raw_text
        state["done"] = True
        trace_item["tool_result"] = None
        trace_item["state_done_after_step"] = True
        state["trace"].append(trace_item)
        return

    state["messages"].append(
        {
            "role": "user",
            "content": (
                "Continue. You must do exactly one of these next (no reasoning text):\n"
                "1) Provide one valid tool call.\n"
                "2) Provide final answer in \\boxed{...}."
            ),
        }
    )
    trace_item["tool_result"] = None
    trace_item["state_done_after_step"] = False
    state["trace"].append(trace_item)


def finalize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    task = deepcopy(state["task"])
    task["rounds"] = state["rounds"]
    task["final_response"] = state.get("final_response", "")
    task["tool_calls"] = int(state.get("tool_calls", 0))
    task["prompt_mode"] = "hard_no_tool"
    task["reasoning_mode"] = "no_reasoning"
    task["trace"] = state.get("trace", [])
    return task


def evaluate_hard_no_tool(tasks: List[Dict[str, Any]], generator: Any, args: argparse.Namespace, tool_format: str) -> List[Dict[str, Any]]:
    states = [initial_state(task, system_prompt_for(tool_format), tool_format) for task in tasks]
    batch_size = max(1, int(args.batch_size))
    for round_idx in range(1, args.max_rounds + 1):
        active = [idx for idx, state in enumerate(states) if not state["done"] and state["rounds"] < args.max_rounds]
        if not active:
            break
        for offset in range(0, len(active), batch_size):
            batch_indices = active[offset : offset + batch_size]
            messages_batch = [states[idx]["messages"] for idx in batch_indices]
            tools_batch = [states[idx]["tools"] for idx in batch_indices]
            outputs = generator.generate_batch(messages_batch, tools_batch)
            for idx, out in zip(batch_indices, outputs):
                step_state(states[idx], out)
        done = sum(1 for state in states if state["done"])
        print(f"round {round_idx}: done={done}/{len(states)}")
    return [finalize_state(state) for state in states]


def evaluate_final_answer(item: Dict[str, Any]) -> Tuple[str, str, str, bool]:
    gold = clean((item.get("expected") or {}).get("answer", ""))
    raw = item.get("final_response", "")
    boxed = extract_boxed(raw)
    model_answer = clean(boxed)
    correct = compare_values(model_answer, gold)
    return raw, boxed, model_answer, correct


def build_label_rows(outputs: List[Dict[str, Any]], model_alias: str, model_path: str, subset: str, split: str) -> List[Dict[str, Any]]:
    rows = []
    for item in outputs:
        raw, boxed, model_answer, correct = evaluate_final_answer(item)
        expected = item.get("expected") or {}
        env_name = item.get("environments", [{}])[0].get("name", "")
        rows.append(
            {
                "model_alias": model_alias,
                "model_path": model_path,
                "source_dataset": "When2Tool",
                "subset": subset,
                "split": split,
                "sample_uid": item.get("sample_uid", f"{subset}:{split}:{item.get('id')}"),
                "id": item.get("id"),
                "difficulty": item.get("difficulty"),
                "multi_step": bool(item.get("multi_step")),
                "env_name": env_name,
                "task_type": item.get("task_type"),
                "task_type_name": item.get("task_type_name"),
                "when2tool_category": item.get("when2tool_category"),
                "answer": expected.get("answer", ""),
                "model_answer_raw": raw,
                "model_answer_boxed": boxed,
                "model_answer": model_answer,
                "no_tool_correct": int(correct),
                "tool_necessary": 0 if correct else 1,
                "rounds": item.get("rounds", 0),
                "tool_calls_rejected": sum(
                    1
                    for trace in item.get("trace", [])
                    if (trace.get("parsed_output") or {}).get("type") == "tool"
                ),
                "prompt_mode": "hard_no_tool",
                "reasoning_mode": "no_reasoning",
            }
        )
    return rows


def summarize_labels(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    correct = sum(int(row["no_tool_correct"]) for row in rows)
    necessary = sum(int(row["tool_necessary"]) for row in rows)
    summary: Dict[str, Any] = {
        "n": n,
        "no_tool_correct": correct,
        "tool_necessary": necessary,
        "no_tool_accuracy": correct / n if n else 0.0,
        "tool_necessary_rate": necessary / n if n else 0.0,
    }
    for key in ["difficulty", "env_name", "task_type"]:
        grouped: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            name = str(row.get(key) or "unknown")
            bucket = grouped.setdefault(name, {"n": 0, "no_tool_correct": 0, "tool_necessary": 0})
            bucket["n"] += 1
            bucket["no_tool_correct"] += int(row["no_tool_correct"])
            bucket["tool_necessary"] += int(row["tool_necessary"])
        for bucket in grouped.values():
            bucket["no_tool_accuracy"] = bucket["no_tool_correct"] / bucket["n"] if bucket["n"] else 0.0
            bucket["tool_necessary_rate"] = bucket["tool_necessary"] / bucket["n"] if bucket["n"] else 0.0
        summary[f"by_{key}"] = dict(sorted(grouped.items()))
    return summary


def run_split(
    model_alias: str,
    model_path: str,
    subset: str,
    split: str,
    raw_dir: Path,
    model_output_root: Path,
    generator: Any,
    args: argparse.Namespace,
    tool_format: str,
) -> Dict[str, Any]:
    tasks = load_raw_split(raw_dir, subset, split, max_samples=args.max_samples)
    print(f"Loaded {len(tasks)} raw When2Tool tasks: {subset}/{split}")
    outputs = evaluate_hard_no_tool(tasks, generator, args, tool_format)
    rows = build_label_rows(outputs, model_alias, model_path, subset, split)
    summary = summarize_labels(rows)

    split_dir = model_output_root / subset / split
    if split_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {split_dir}. Use --overwrite.")
    split_dir.mkdir(parents=True, exist_ok=True)
    write_json(split_dir / "no_tool_outputs.json", outputs)
    write_jsonl(split_dir / "labels.jsonl", rows)
    write_json(split_dir / "summary.json", summary)
    print(f"Saved labels: {split_dir / 'labels.jsonl'}")
    print(f"Summary: no_tool_accuracy={summary['no_tool_accuracy']:.4f}, tool_necessary_rate={summary['tool_necessary_rate']:.4f}")
    return {"subset": subset, "split": split, "output_dir": str(split_dir), **summary}


def run_model(model_alias_or_path: str, args: argparse.Namespace) -> None:
    model_spec = resolve_model_spec(
        args.model_path or model_alias_or_path,
        config_path=args.models_config,
        prefer_local=args.prefer_local and not bool(args.model_path),
    )
    model_alias = normalize_model_key(model_alias_or_path)
    if args.model_path:
        model_alias = normalize_model_key(args.model_alias)
    data_root = Path(args.data_root)
    raw_dir = Path(args.raw_dir) if args.raw_dir else data_root / "datasets" / "raw_when2tool"
    output_root = Path(args.output_root) if args.output_root else data_root / "labels"
    model_output_root = output_root / model_alias

    print("=" * 80)
    print(f"MODEL: {model_alias}")
    print(f"repo_id: {model_spec.repo_id}")
    print(f"resolved_path: {model_spec.resolved_path}")
    print(f"resolved_from_local: {model_spec.resolved_from_local}")
    print(f"backend: {args.backend}")
    print("=" * 80)

    tool_format = detect_tool_format(model_spec.repo_id)
    generator = build_generator(model_spec.resolved_path, args)
    split_summaries = []
    for subset in args.subsets:
        for split in args.splits:
            split_summaries.append(
                run_split(
                    model_alias=model_alias,
                    model_path=model_spec.resolved_path,
                    subset=subset,
                    split=split,
                    raw_dir=raw_dir,
                    model_output_root=model_output_root,
                    generator=generator,
                    args=args,
                    tool_format=tool_format,
                )
            )

    manifest = {
        "model_alias": model_alias,
        "repo_id": model_spec.repo_id,
        "resolved_path": model_spec.resolved_path,
        "resolved_from_local": model_spec.resolved_from_local,
        "input_dataset": str(raw_dir),
        "output_root": str(model_output_root),
        "label_definition": "tool_necessary = 0 if hard_no_tool final answer is correct else 1",
        "prompt_mode": "hard_no_tool",
        "reasoning_mode": "no_reasoning",
        "backend": args.backend,
        "max_rounds": args.max_rounds,
        "max_new_tokens": args.max_new_tokens,
        "splits": split_summaries,
    }
    write_json(model_output_root / "manifest.json", manifest)
    print(f"Saved model manifest: {model_output_root / 'manifest.json'}")


def check_data_only(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    raw_dir = Path(args.raw_dir) if args.raw_dir else data_root / "datasets" / "raw_when2tool"
    print(f"Checking raw When2Tool data: {raw_dir}")
    for subset in args.subsets:
        for split in args.splits:
            tasks = load_raw_split(raw_dir, subset, split, max_samples=args.max_samples)
            for task in tasks:
                build_tools_schema(task)
            counts = pd.Series([task["task_type"] for task in tasks]).value_counts().sort_index().to_dict()
            print(f"{subset}/{split}: {len(tasks)} tasks, task_type_counts={counts}")
    print("Data/schema check passed.")


def main() -> None:
    args = parse_args()
    if args.check_data_only:
        check_data_only(args)
        return
    aliases: Sequence[str]
    if normalize_model_key(args.model_alias) == "all":
        aliases = list(list_model_aliases(args.models_config))
    else:
        aliases = [args.model_alias]
    for alias in aliases:
        run_model(alias, args)


if __name__ == "__main__":
    main()
