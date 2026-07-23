#!/usr/bin/env python3
"""Stage 6 single-type causal validation.

It follows the stage-6 plan:
evaluate Base, same-shape Random masks, and TDN masks on A/B/C test tasks.

Neuron definition follows Who Transfers Safety:
- Q/K/V neurons are rows of W_Q/W_K/W_V, implemented by zeroing the matching
  projection-output coordinates during forward.
- O neurons are columns of W_O, implemented by zeroing the matching o_proj
  input coordinates during forward.
"""

from __future__ import annotations

import argparse
import ast
import csv
import importlib
import json
import math
import random
import re
import signal
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import io
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
for path in (str(CODE_ROOT),):
    if path not in sys.path:
        sys.path.insert(0, path)

from common.io_utils import write_json, write_jsonl  # noqa: E402
from common import causal_plots  # noqa: E402

TASK_TYPES = ("A", "B", "C")
SUBSETS = ("single_hop", "multi_hop")
MATRICES = ("Q", "K", "V", "O")

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-alias", default="qwen3-4b-instruct")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--modified-dir", required=True)
    parser.add_argument("--neuron-dir", required=True, help="Stage-5 neuron dir for one model.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-types", nargs="+", default=list(TASK_TYPES), choices=TASK_TYPES)
    parser.add_argument("--subsets", nargs="+", default=list(SUBSETS), choices=SUBSETS)
    parser.add_argument(
        "--separate-subsets",
        action="store_true",
        help="Use neurons from single_type_by_subset/<subset> and evaluate each subset independently.",
    )
    parser.add_argument("--split", default="test")
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
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--refresh-existing",
        action="store_true",
        help="Recompute tables, manifests, and figures from existing per_task.jsonl files without loading the model.",
    )
    return parser.parse_args()


def torch_dtype(name: str) -> Any:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    return "auto"


def enable_thinking_value(value: str) -> Optional[bool]:
    if value == "true":
        return True
    if value == "false":
        return False
    return None


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
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


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


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def detect_tool_format(model_name_or_path: str) -> str:
    return "native" if "llama" in str(model_name_or_path).lower() else "xml"


def system_prompt_for(tool_format: str) -> str:
    return SYSTEM_PROMPT_NATIVE if tool_format == "native" else SYSTEM_PROMPT_XML


def build_current_no_reasoning_user_message(task_instruction: str) -> str:
    return (
        task_instruction
        + "\n\nResponse policy (required every turn):\n"
        + "1) You can choose to use a tool or not in this task.\n"
        + "2) Provide final answer in \\boxed{...} if you think the task is complete."
    )


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


def modified_record_to_task(record: Dict[str, Any]) -> Dict[str, Any]:
    tools = parse_json_field(record.get("tools"), [])
    parameters = parse_json_field(record.get("parameters"), {})
    steps = parse_json_field(record.get("steps"), [])
    expected = {"answer": record.get("answer", "")}
    if steps:
        expected["steps"] = steps
    return {
        "id": record.get("id"),
        "sample_uid": record.get("sample_uid", f"{record.get('subset')}:{record.get('split')}:{record.get('id')}"),
        "source_dataset": "When2Tool",
        "subset": record.get("subset"),
        "split": record.get("split"),
        "difficulty": record.get("difficulty"),
        "multi_step": bool(record.get("multi_step")),
        "instruction": str(record.get("instruction", "")),
        "environments": [{"name": record["env_name"], "tools": tools, "parameters": parameters}],
        "expected": expected,
        "tags": parse_json_field(record.get("tags"), []),
        "env_name": record["env_name"],
        "task_type": record["task_type"],
        "task_type_name": record.get("task_type_name"),
        "when2tool_category": record.get("when2tool_category"),
        "tool_necessary": int(record.get("tool_necessary", 0)),
        "no_tool_correct": int(record.get("no_tool_correct", 0)),
    }


def load_test_tasks(modified_dir: Path, subsets: Iterable[str], split: str, task_types: Iterable[str], max_per_type: int) -> Dict[str, List[Dict[str, Any]]]:
    grouped = {task_type: [] for task_type in task_types}
    for subset in subsets:
        path = modified_dir / subset / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing modified split: {path}")
        for record in read_jsonl(path):
            task_type = str(record.get("task_type"))
            if task_type in grouped:
                grouped[task_type].append(modified_record_to_task(record))
    for task_type, tasks in grouped.items():
        tasks.sort(key=lambda row: (str(row.get("subset")), int(row.get("id") or 0)))
        if max_per_type > 0:
            grouped[task_type] = tasks[:max_per_type]
    return grouped


# ---------------------------------------------------------------------------
# Official When2Tool environment adapter.
# ---------------------------------------------------------------------------


ENV_CLASS_IMPORTS = {
    "CalculatorEnv": ("third_party.when2tool_adapter.envs.calculator_env", "CalculatorEnv"),
    "StatisticsEnv": ("third_party.when2tool_adapter.envs.statistics_env", "StatisticsEnv"),
    "CountingEnv": ("third_party.when2tool_adapter.envs.counting_env", "CountingEnv"),
    "MatrixEnv": ("third_party.when2tool_adapter.envs.matrix_env", "MatrixEnv"),
    "PrimeEnv": ("third_party.when2tool_adapter.envs.prime_env", "PrimeEnv"),
    "RetrieverEnv": ("third_party.when2tool_adapter.envs.retriever_env", "RetrieverEnv"),
    "HistoricalYearEnv": ("third_party.when2tool_adapter.envs.historical_year_env", "HistoricalYearEnv"),
    "GameRuleEnv": ("third_party.when2tool_adapter.envs.game_rule_env", "GameRuleEnv"),
    "HashEnv": ("third_party.when2tool_adapter.envs.hash_env", "HashEnv"),
    "DecodingEnv": ("third_party.when2tool_adapter.envs.decoding_env", "DecodingEnv"),
    "ListManipulationEnv": ("third_party.when2tool_adapter.envs.list_manipulation_env", "ListManipulationEnv"),
    "DateTimeEnv": ("third_party.when2tool_adapter.envs.datetime_env", "DateTimeEnv"),
    "CodeExecutorEnv": ("third_party.when2tool_adapter.envs.code_executor_env", "CodeExecutorEnv"),
    "ScheduleEnv": ("third_party.when2tool_adapter.envs.schedule_env", "ScheduleEnv"),
    "RegexMatchEnv": ("third_party.when2tool_adapter.envs.regex_match_env", "RegexMatchEnv"),
}


_ENV_CLASS_CACHE: Dict[str, Any] = {}


def get_env_class(env_name: str) -> Any:
    if env_name not in ENV_CLASS_IMPORTS:
        raise ValueError(f"When2Tool env not implemented: {env_name}")
    if env_name not in _ENV_CLASS_CACHE:
        module_name, class_name = ENV_CLASS_IMPORTS[env_name]
        module = importlib.import_module(module_name)
        _ENV_CLASS_CACHE[env_name] = getattr(module, class_name)
    return _ENV_CLASS_CACHE[env_name]


def load_env_schemas(env_name: str) -> List[Dict[str, Any]]:
    env = get_env_class(env_name)()
    return deepcopy(env.tool_descs)


def build_envs(task: Dict[str, Any]) -> Tuple[List[Tuple[Any, set]], List[Dict[str, Any]]]:
    envs = []
    tools = []
    for env_cfg in task.get("environments", []):
        env_name = env_cfg["name"]
        env_cls = get_env_class(env_name)
        env = env_cls(parameters=deepcopy(env_cfg.get("parameters") or {}))
        allowed_tools = set(env_cfg.get("tools") or [])
        envs.append((env, allowed_tools))
        for schema in env.tool_descs:
            if schema["name"] in allowed_tools:
                tools.append({"type": "function", "function": deepcopy(schema)})
    return envs, tools


def route_tool_call(envs: List[Tuple[Any, set]], tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    for env, allowed in envs:
        if tool_name in allowed and env.has_tool(tool_name):
            return env.call_tool(tool_name, arguments or {})
    return {"success": False, "message": f"Tool {tool_name} not available in this task."}


# ---------------------------------------------------------------------------
# Scoring and generation parsing.
# ---------------------------------------------------------------------------


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def extract_boxed(text: Any) -> str:
    value = clean(text)
    marker = "\\boxed{"
    idx = value.find(marker)
    if idx < 0:
        return ""
    pos = idx + len(marker)
    depth = 1
    out: List[str] = []
    while pos < len(value):
        char = value[pos]
        if char == "{":
            depth += 1
            out.append(char)
        elif char == "}":
            depth -= 1
            if depth == 0:
                break
            out.append(char)
        else:
            out.append(char)
        pos += 1
    return "".join(out).strip() if out else value


def normalize_scalar(value: Any) -> str:
    text = re.sub(r"\s+", " ", clean(value)).strip()
    if len(text) >= 2 and ((text[0] == "'" and text[-1] == "'") or (text[0] == '"' and text[-1] == '"')):
        text = text[1:-1].strip()
    text = re.sub(r"\\text\s*\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\mathrm\s*\{([^}]*)\}", r"\1", text)
    text = text.replace("{", "").replace("}", "").replace("\\", "")
    return re.sub(r"\s+", " ", text).strip().lower()


def normalize_structured(value: Any) -> Any:
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return [normalize_structured(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize_structured(item) for key, item in value.items()}
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


def has_reasoning_before_tool(raw_text: Any) -> bool:
    text = str(raw_text or "")
    if not text.strip():
        return False
    idx = text.find("<tool_call>")
    if idx >= 0:
        return _has_nontrivial_reasoning(text[:idx])
    first_json = text.find("{")
    if first_json <= 0:
        return False
    return _has_nontrivial_reasoning(text[:first_json])


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
        return {"type": "tool", "tool_name": parsed["name"], "arguments": parsed.get("arguments", {}), "raw_text": raw_text}
    return {"type": "content", "content": raw_text, "raw_text": raw_text}


def item_expected_steps(item: Dict[str, Any]) -> int:
    steps = (item.get("expected") or {}).get("steps")
    if isinstance(steps, list) and len(steps) > 0:
        return len(steps)
    return 3 if bool(item.get("multi_step")) else 1


def evaluate_final_answer(item: Dict[str, Any]) -> Tuple[str, str, str, bool]:
    raw = item.get("final_response", "")
    boxed = extract_boxed(raw)
    model_answer = clean(boxed)
    gold = clean((item.get("expected") or {}).get("answer", ""))
    return raw, boxed, model_answer, compare_values(model_answer, gold)


# ---------------------------------------------------------------------------
# Causal masks.
# ---------------------------------------------------------------------------


def layer_modules(model: Any) -> List[Any]:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise AttributeError("Cannot find model.model.layers; unsupported architecture.")
    return list(layers)


def neuron_key(row: Dict[str, Any]) -> Tuple[int, str, int]:
    return int(row["layer"]), str(row["matrix"]), int(row["index"])


def read_tdn_rows(neuron_dir: Path, task_type: str) -> List[Dict[str, Any]]:
    path = neuron_dir / task_type / "TDN_neurons.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing TDN file: {path}")
    return read_jsonl(path)


def load_matrix_dims(neuron_dir: Path) -> Dict[str, Tuple[int, int]]:
    manifest = json.loads((neuron_dir / "manifest.json").read_text(encoding="utf-8"))
    return {matrix: (int(shape[0]), int(shape[1])) for matrix, shape in manifest["matrix_dims"].items()}


def rows_to_mask(rows: List[Dict[str, Any]]) -> Dict[Tuple[int, str], List[int]]:
    grouped: Dict[Tuple[int, str], List[int]] = defaultdict(list)
    for row in rows:
        layer, matrix, index = neuron_key(row)
        grouped[(layer, matrix)].append(index)
    return {key: sorted(set(values)) for key, values in grouped.items()}


def random_like_mask(tdn_rows: List[Dict[str, Any]], dims: Dict[str, Tuple[int, int]], seed: int, task_type: str) -> Dict[Tuple[int, str], List[int]]:
    rng = random.Random(f"{seed}:{task_type}:random_same_shape")
    counts = Counter((int(row["layer"]), str(row["matrix"])) for row in tdn_rows)
    tdn_by_group = rows_to_mask(tdn_rows)
    out: Dict[Tuple[int, str], List[int]] = {}
    for (layer, matrix), count in sorted(counts.items()):
        dim = dims[matrix][1]
        excluded = set(tdn_by_group.get((layer, matrix), []))
        pool = [idx for idx in range(dim) if idx not in excluded]
        if len(pool) < count:
            pool = list(range(dim))
        out[(layer, matrix)] = sorted(rng.sample(pool, count))
    return out


def mask_size(mask: Dict[Tuple[int, str], List[int]]) -> int:
    return sum(len(values) for values in mask.values())


class CausalHFGenerator:
    def __init__(self, model_path: str, args: argparse.Namespace):
        self.model_path = model_path
        self.max_new_tokens = args.max_new_tokens
        self.enable_thinking = enable_thinking_value(args.enable_thinking)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype(args.torch_dtype),
            device_map=args.device_map,
            local_files_only=True,
        ).eval()
        self.generation_config = GenerationConfig.from_pretrained(model_path, local_files_only=True)
        if args.temperature is not None:
            self.generation_config.temperature = float(args.temperature)
        if args.top_p is not None:
            self.generation_config.top_p = float(args.top_p)
        if args.top_k is not None:
            self.generation_config.top_k = int(args.top_k)

    def render_prompt(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]]) -> str:
        kwargs: Dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if tools:
            kwargs["tools"] = tools
        if self.enable_thinking is not None:
            kwargs["enable_thinking"] = self.enable_thinking
        try:
            return self.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    @contextmanager
    def masked(self, mask: Optional[Dict[Tuple[int, str], List[int]]]):
        hooks = []
        if mask:
            for layer_idx, layer in enumerate(layer_modules(self.model)):
                attn = layer.self_attn

                def save_output(matrix: str, idx: int):
                    indices = mask.get((idx, matrix), [])

                    def hook(_module: Any, _inputs: Tuple[Any, ...], output: Any) -> Any:
                        if not indices:
                            return output
                        tensor = output[0] if isinstance(output, tuple) else output
                        edited = tensor.clone()
                        edited[..., torch.tensor(indices, device=edited.device)] = 0
                        if isinstance(output, tuple):
                            return (edited,) + tuple(output[1:])
                        return edited

                    return hook

                def save_input(idx: int):
                    indices = mask.get((idx, "O"), [])

                    def hook(_module: Any, inputs: Tuple[Any, ...]) -> Tuple[Any, ...]:
                        if not indices:
                            return inputs
                        edited = inputs[0].clone()
                        edited[..., torch.tensor(indices, device=edited.device)] = 0
                        return (edited,) + tuple(inputs[1:])

                    return hook

                hooks.append(attn.q_proj.register_forward_hook(save_output("Q", layer_idx)))
                hooks.append(attn.k_proj.register_forward_hook(save_output("K", layer_idx)))
                hooks.append(attn.v_proj.register_forward_hook(save_output("V", layer_idx)))
                hooks.append(attn.o_proj.register_forward_pre_hook(save_input(layer_idx)))
        try:
            yield
        finally:
            for hook in hooks:
                hook.remove()

    def generate_batch(
        self,
        messages_batch: List[List[Dict[str, str]]],
        tools_batch: List[List[Dict[str, Any]]],
        mask: Optional[Dict[Tuple[int, str], List[int]]] = None,
    ) -> List[Dict[str, Any]]:
        outputs = []
        embed_device = self.model.get_input_embeddings().weight.device
        with self.masked(mask):
            for messages, tools in zip(messages_batch, tools_batch):
                prompt_text = self.render_prompt(messages, tools)
                inputs = self.tokenizer(prompt_text, return_tensors="pt").to(embed_device)
                generated = self.model.generate(
                    **inputs,
                    generation_config=deepcopy(self.generation_config),
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                raw_text = self.tokenizer.decode(generated[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True)
                out = normalize_generation_output(raw_text)
                out["prompt_text"] = prompt_text
                out["finish_reason"] = "length_or_eos"
                outputs.append(out)
        return outputs


def initial_state(task: Dict[str, Any], system_prompt: str, tool_format: str, record_mode: str) -> Dict[str, Any]:
    envs, tools = build_envs(task)
    messages = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "user", "content": build_current_no_reasoning_user_message(task["instruction"])})
    return {
        "task": deepcopy(task),
        "envs": envs,
        "tools": tools,
        "messages": messages,
        "rounds": 0,
        "done": False,
        "trace": [],
        "final_response": "",
        "tool_calls_used": 0,
        "tool_format": tool_format,
        "record_mode": record_mode,
    }


def step_state(state: Dict[str, Any], out: Dict[str, Any]) -> None:
    state["rounds"] += 1
    raw_text = out.get("raw_text", out.get("content", ""))
    state["messages"].append({"role": "assistant", "content": raw_text})
    trace_item = {
        "round": state["rounds"],
        "prompt_text": out.get("prompt_text", "") if state["record_mode"] in {"lite", "full"} else "",
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
        if not state.get("_last_tool_success") and has_reasoning_before_tool(raw_text):
            state["messages"].append(
                {
                    "role": "user",
                    "content": "Tool call rejected: reasoning is not allowed in no_reasoning mode. Retry with tool call only.",
                }
            )
            trace_item["tool_result"] = {"success": False, "message": "Tool call rejected: reasoning is not allowed in no_reasoning mode."}
            trace_item["state_done_after_step"] = False
            state["trace"].append(trace_item)
            return
        tool_result = route_tool_call(state["envs"], out["tool_name"], deepcopy(out.get("arguments", {})))
        state["tool_calls_used"] += 1
        content = json.dumps(tool_result, ensure_ascii=False)
        if state["tool_format"] == "native":
            state["messages"].append({"role": "tool", "content": content})
        else:
            state["messages"].append({"role": "user", "content": "<tool_response>\n" + content + "\n</tool_response>"})
        trace_item["tool_result"] = deepcopy(tool_result)
        trace_item["state_done_after_step"] = False
        state["trace"].append(trace_item)
        state["_last_tool_success"] = True
        return

    if has_boxed_answer(raw_text):
        if not state.get("_last_tool_success") and has_reasoning_for_direct_answer(raw_text):
            state["messages"].append(
                {
                    "role": "user",
                    "content": "Final answer rejected: reasoning is not allowed in no_reasoning mode. Retry with final answer in \\boxed{...} only.",
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

    state.pop("_last_tool_success", None)
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
    item = deepcopy(state["task"])
    item["rounds"] = int(state["rounds"])
    item["final_response"] = state.get("final_response", "")
    item["tool_calls"] = int(state.get("tool_calls_used", 0))
    item["prompt_mode"] = "current"
    item["reasoning_mode"] = "no_reasoning"
    if state.get("record_mode") != "off":
        item["trace"] = state.get("trace", [])
    return item


def evaluate_tasks(
    tasks: List[Dict[str, Any]],
    generator: CausalHFGenerator,
    mask: Optional[Dict[Tuple[int, str], List[int]]],
    max_rounds: int,
    tool_format: str,
    record_mode: str,
) -> List[Dict[str, Any]]:
    states = [initial_state(task, system_prompt_for(tool_format), tool_format, record_mode) for task in tasks]
    for round_idx in range(1, max_rounds + 1):
        active = [idx for idx, state in enumerate(states) if not state["done"] and state["rounds"] < max_rounds]
        if not active:
            break
        messages_batch = [states[idx]["messages"] for idx in active]
        tools_batch = [states[idx]["tools"] for idx in active]
        outputs = generator.generate_batch(messages_batch, tools_batch, mask=mask)
        for idx, out in zip(active, outputs):
            step_state(states[idx], out)
        done = sum(1 for state in states if state["done"])
        print(f"round {round_idx}: active={len(active)} done={done}/{len(states)}")
    return [finalize_state(state) for state in states]


def per_task_row(item: Dict[str, Any], model_alias: str, task_type: str, intervention: str, mask_count: int) -> Dict[str, Any]:
    raw, boxed, model_answer, final_correct = evaluate_final_answer(item)
    tool_calls = max(0, int(item.get("tool_calls", 0)))
    expected_steps = item_expected_steps(item)
    y = int(item.get("tool_necessary", 0))
    used_tool = int(tool_calls > 0)
    return {
        "model_alias": model_alias,
        "task_type": task_type,
        "intervention": intervention,
        "mask_count": mask_count,
        "subset": item.get("subset"),
        "split": item.get("split"),
        "sample_uid": item.get("sample_uid"),
        "id": item.get("id"),
        "env_name": item.get("env_name"),
        "difficulty": item.get("difficulty"),
        "tool_necessary": y,
        "used_tool": used_tool,
        "tool_decision_correct": int(used_tool == y),
        "final_correct": int(final_correct),
        "tool_calls": tool_calls,
        "expected_steps": expected_steps,
        "tool_call_rate": tool_calls / expected_steps if expected_steps else 0.0,
        "model_answer_raw": raw,
        "model_answer_boxed": boxed,
        "model_answer": model_answer,
        "answer": (item.get("expected") or {}).get("answer", ""),
        "rounds": item.get("rounds", 0),
    }


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    correct = sum(int(row["final_correct"]) for row in rows)
    tool_calls = sum(int(row["tool_calls"]) for row in rows)
    expected_steps = sum(int(row["expected_steps"]) for row in rows)
    tool_decision_correct = sum(int(row["tool_decision_correct"]) for row in rows)
    y1 = [row for row in rows if int(row["tool_necessary"]) == 1]
    y0 = [row for row in rows if int(row["tool_necessary"]) == 0]
    tool_necessary_correct = sum(int(row["used_tool"]) for row in y1)
    no_tool_correct = sum(int(not row["used_tool"]) for row in y0)
    tool_necessary_accuracy = tool_necessary_correct / len(y1) if y1 else 0.0
    no_tool_accuracy = no_tool_correct / len(y0) if y0 else 0.0
    overcall = (sum(int(row["used_tool"]) for row in y0) / len(y0)) if y0 else 0.0
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "tool_calls": tool_calls,
        "avg_tool_calls": tool_calls / n if n else 0.0,
        "tool_call_rate": tool_calls / expected_steps if expected_steps else 0.0,
        "expected_steps": expected_steps,
        "tool_decision_correct": tool_decision_correct,
        "tool_accuracy": tool_decision_correct / n if n else 0.0,
        "tool_necessary_correct": tool_necessary_correct,
        "tool_necessary_accuracy": tool_necessary_accuracy,
        "no_tool_correct": no_tool_correct,
        "no_tool_accuracy": no_tool_accuracy,
        "recall_tool": tool_necessary_accuracy,
        "overcall": overcall,
        "n_tool_necessary_1": len(y1),
        "n_tool_necessary_0": len(y0),
    }


def markdown_table(summary_rows: List[Dict[str, Any]]) -> str:
    lines = [
        "# Stage 6 Single-Type Causal Validation",
        "",
        "| Model | Subset | Type | Intervention | MaskN | N | Acc | TC | AvgTC | TCR | ToolAcc | ToolNecessaryAcc | NoToolAcc | OverCall |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['model_alias']} | {row.get('subset_scope', 'combined')} | {row['task_type']} | {row['intervention']} | "
            f"{row['mask_count']} | {row['n']} | {row['accuracy']:.4f} | {row['tool_calls']} | "
            f"{row['avg_tool_calls']:.4f} | {row['tool_call_rate']:.4f} | {row['tool_accuracy']:.4f} | "
            f"{row['tool_necessary_accuracy']:.4f} | {row['no_tool_accuracy']:.4f} | {row['overcall']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Metric notes:",
            "- Acc / TC / AvgTC / TCR follow When2Tool-style final-answer and tool-call metrics.",
            "- ToolAcc is tool-decision accuracy: whether used_tool equals tool_necessary.",
            "- ToolNecessaryAcc is accuracy on tool_necessary=1 samples.",
            "- NoToolAcc is accuracy on tool_necessary=0 samples.",
            "- OverCall is the tool-use rate on tool_necessary=0 samples, so NoToolAcc = 1 - OverCall when such samples exist.",
            "",
        ]
    )
    return "\n".join(lines)


def run_one(
    generator: CausalHFGenerator,
    tasks: List[Dict[str, Any]],
    model_alias: str,
    subset_scope: str,
    task_type: str,
    intervention: str,
    mask: Optional[Dict[Tuple[int, str], List[int]]],
    max_rounds: int,
    tool_format: str,
    record_mode: str,
    output_root: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    print(f"=== {subset_scope} / {task_type} / {intervention}: tasks={len(tasks)}, mask={mask_size(mask or {})} ===")
    outputs = evaluate_tasks(tasks, generator, mask, max_rounds, tool_format, record_mode)
    run_dir = output_root / task_type / intervention
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "outputs.json", outputs)
    rows = [per_task_row(item, model_alias, task_type, intervention, mask_size(mask or {})) for item in outputs]
    write_jsonl(run_dir / "per_task.jsonl", rows)
    summary = summarize_rows(rows)
    summary.update(
        {
            "model_alias": model_alias,
            "subset_scope": subset_scope,
            "task_type": task_type,
            "intervention": intervention,
            "mask_count": mask_size(mask or {}),
        }
    )
    write_json(run_dir / "summary.json", summary)
    return rows, summary


SUMMARY_COLUMNS = [
    "model_alias",
    "subset_scope",
    "task_type",
    "intervention",
    "mask_count",
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
    "recall_tool",
    "overcall",
    "n_tool_necessary_1",
    "n_tool_necessary_0",
]


def run_scope(
    args: argparse.Namespace,
    modified_dir: Path,
    neuron_dir: Path,
    output_root: Path,
    subset_scope: str,
    subsets: List[str],
    generator: CausalHFGenerator,
    tool_format: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    output_root.mkdir(parents=True, exist_ok=True)
    tasks_by_type = load_test_tasks(modified_dir, subsets, args.split, args.task_types, args.max_tasks_per_type)
    for task_type, tasks in tasks_by_type.items():
        print(f"{subset_scope}/{task_type}: loaded {len(tasks)} {args.split} tasks")
        envs = sorted({task["env_name"] for task in tasks})
        print(f"{subset_scope}/{task_type}: envs={envs}")

    dims = load_matrix_dims(neuron_dir)
    tdn_rows_by_type = {task_type: read_tdn_rows(neuron_dir, task_type) for task_type in args.task_types}
    tdn_masks = {task_type: rows_to_mask(rows) for task_type, rows in tdn_rows_by_type.items()}
    random_masks = {
        task_type: random_like_mask(tdn_rows_by_type[task_type], dims, args.seed, f"{subset_scope}:{task_type}")
        for task_type in args.task_types
    }

    all_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for task_type in args.task_types:
        tasks = tasks_by_type[task_type]
        runs = [
            ("Base", None),
            ("M-Random", random_masks[task_type]),
            (f"M-TDN_{task_type}", tdn_masks[task_type]),
        ]
        for intervention, mask in runs:
            rows, summary = run_one(
                generator,
                tasks,
                args.model_alias,
                subset_scope,
                task_type,
                intervention,
                mask,
                args.max_rounds,
                tool_format,
                args.record_mode,
                output_root,
            )
            all_rows.extend(rows)
            summary_rows.append(summary)

    write_csv(output_root / "summary_table.csv", summary_rows, SUMMARY_COLUMNS)
    (output_root / "summary.md").write_text(markdown_table(summary_rows), encoding="utf-8")
    write_jsonl(output_root / "all_per_task.jsonl", all_rows)
    figures = causal_plots.plot_stage6_single_type(output_root, neuron_dir, summary_rows)
    write_json(
        output_root / "manifest.json",
        {
            "stage": "stage6_single_type_causal_validation_scope",
            "model_alias": args.model_alias,
            "model_path": args.model_path,
            "modified_dir": str(modified_dir),
            "neuron_dir": str(neuron_dir),
            "output_dir": str(output_root),
            "split": args.split,
            "subsets": list(subsets),
            "subset_scope": subset_scope,
            "task_types": list(args.task_types),
            "prompt_mode": "current",
            "reasoning_mode": "no_reasoning",
            "max_rounds": args.max_rounds,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "neuron_definition": "Who Transfers Safety attention projection neurons: Q/K/V projection rows and O output-projection columns.",
            "mask_implementation": "Q/K/V zero projection-output coordinates; O zero o_proj input coordinates, equivalent to masking W_O columns.",
            "metrics": {
                "Acc": "final-answer accuracy, aligned with When2Tool",
                "TC": "total tool calls",
                "AvgTC": "average tool calls per task",
                "TCR": "tool calls divided by expected steps",
                "ToolAcc": "mean 1[used_tool == tool_necessary]",
                "ToolNecessaryAcc": "mean used_tool on tool_necessary=1 samples",
                "NoToolAcc": "mean 1[not used_tool] on tool_necessary=0 samples",
                "OverCall": "tool-use rate on tool_necessary=0 samples",
            },
            "figures": figures,
            "summary_rows": summary_rows,
        },
    )
    return all_rows, summary_rows


def refresh_scope(
    args: argparse.Namespace,
    modified_dir: Path,
    neuron_dir: Path,
    output_root: Path,
    subset_scope: str,
    subsets: List[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    output_root.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for task_type in args.task_types:
        interventions = ["Base", "M-Random", f"M-TDN_{task_type}"]
        for intervention in interventions:
            run_dir = output_root / task_type / intervention
            rows = read_jsonl(run_dir / "per_task.jsonl")
            old_summary = read_json(run_dir / "summary.json") if (run_dir / "summary.json").exists() else {}
            summary = summarize_rows(rows)
            summary.update(
                {
                    "model_alias": old_summary.get("model_alias", args.model_alias),
                    "subset_scope": old_summary.get("subset_scope", subset_scope),
                    "task_type": old_summary.get("task_type", task_type),
                    "intervention": old_summary.get("intervention", intervention),
                    "mask_count": old_summary.get("mask_count", rows[0].get("mask_count", 0) if rows else 0),
                }
            )
            write_json(run_dir / "summary.json", summary)
            all_rows.extend(rows)
            summary_rows.append(summary)

    write_csv(output_root / "summary_table.csv", summary_rows, SUMMARY_COLUMNS)
    (output_root / "summary.md").write_text(markdown_table(summary_rows), encoding="utf-8")
    write_jsonl(output_root / "all_per_task.jsonl", all_rows)
    figures = causal_plots.plot_stage6_single_type(output_root, neuron_dir, summary_rows)
    write_json(
        output_root / "manifest.json",
        {
            "stage": "stage6_single_type_causal_validation_scope",
            "refreshed_from_existing_per_task": True,
            "model_alias": args.model_alias,
            "model_path": args.model_path,
            "modified_dir": str(modified_dir),
            "neuron_dir": str(neuron_dir),
            "output_dir": str(output_root),
            "split": args.split,
            "subsets": list(subsets),
            "subset_scope": subset_scope,
            "task_types": list(args.task_types),
            "prompt_mode": "current",
            "reasoning_mode": "no_reasoning",
            "max_rounds": args.max_rounds,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "neuron_definition": "Who Transfers Safety attention projection neurons: Q/K/V projection rows and O output-projection columns.",
            "mask_implementation": "Q/K/V zero projection-output coordinates; O zero o_proj input coordinates, equivalent to masking W_O columns.",
            "metrics": {
                "Acc": "final-answer accuracy, aligned with When2Tool",
                "TC": "total tool calls",
                "AvgTC": "average tool calls per task",
                "TCR": "tool calls divided by expected steps",
                "ToolAcc": "mean 1[used_tool == tool_necessary]",
                "ToolNecessaryAcc": "mean used_tool on tool_necessary=1 samples",
                "NoToolAcc": "mean 1[not used_tool] on tool_necessary=0 samples",
                "OverCall": "tool-use rate on tool_necessary=0 samples",
            },
            "figures": figures,
            "summary_rows": summary_rows,
        },
    )
    return all_rows, summary_rows


def main() -> None:
    args = parse_args()
    modified_dir = Path(args.modified_dir)
    neuron_dir = Path(args.neuron_dir)
    output_name = "single_type_by_subset" if args.separate_subsets else "single_type"
    output_root = Path(args.output_dir) / args.model_alias / output_name
    if output_root.exists() and not (args.overwrite or args.refresh_existing):
        raise FileExistsError(f"Output exists: {output_root}. Use --overwrite.")
    output_root.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    if args.refresh_existing:
        if args.separate_subsets:
            for subset in args.subsets:
                scope_rows, scope_summary = refresh_scope(
                    args,
                    modified_dir,
                    neuron_dir / subset,
                    output_root / subset,
                    subset,
                    [subset],
                )
                all_rows.extend(scope_rows)
                summary_rows.extend(scope_summary)
        else:
            scope_rows, scope_summary = refresh_scope(
                args,
                modified_dir,
                neuron_dir,
                output_root,
                "combined",
                list(args.subsets),
            )
            all_rows.extend(scope_rows)
            summary_rows.extend(scope_summary)
    else:
        tool_format = detect_tool_format(args.model_path)
        generator = CausalHFGenerator(args.model_path, args)
        if args.separate_subsets:
            for subset in args.subsets:
                scope_rows, scope_summary = run_scope(
                    args,
                    modified_dir,
                    neuron_dir / subset,
                    output_root / subset,
                    subset,
                    [subset],
                    generator,
                    tool_format,
                )
                all_rows.extend(scope_rows)
                summary_rows.extend(scope_summary)
        else:
            scope_rows, scope_summary = run_scope(
                args,
                modified_dir,
                neuron_dir,
                output_root,
                "combined",
                list(args.subsets),
                generator,
                tool_format,
            )
            all_rows.extend(scope_rows)
            summary_rows.extend(scope_summary)

    write_csv(output_root / "summary_table.csv", summary_rows, SUMMARY_COLUMNS)
    (output_root / "summary.md").write_text(markdown_table(summary_rows), encoding="utf-8")
    write_jsonl(output_root / "all_per_task.jsonl", all_rows)
    write_json(
        output_root / "manifest.json",
        {
            "stage": "stage6_single_type_causal_validation",
            "model_alias": args.model_alias,
            "model_path": args.model_path,
            "modified_dir": str(modified_dir),
            "neuron_dir": str(neuron_dir),
            "output_dir": str(output_root),
            "split": args.split,
            "subsets": list(args.subsets),
            "separated_by_hop": args.separate_subsets,
            "task_types": list(args.task_types),
            "prompt_mode": "current",
            "reasoning_mode": "no_reasoning",
            "max_rounds": args.max_rounds,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "neuron_definition": "Who Transfers Safety attention projection neurons: Q/K/V projection rows and O output-projection columns.",
            "mask_implementation": "Q/K/V zero projection-output coordinates; O zero o_proj input coordinates, equivalent to masking W_O columns.",
            "metrics": {
                "Acc": "final-answer accuracy, aligned with When2Tool",
                "TC": "total tool calls",
                "AvgTC": "average tool calls per task",
                "TCR": "tool calls divided by expected steps",
                "ToolAcc": "mean 1[used_tool == tool_necessary]",
                "ToolNecessaryAcc": "mean used_tool on tool_necessary=1 samples",
                "NoToolAcc": "mean 1[not used_tool] on tool_necessary=0 samples",
                "OverCall": "tool-use rate on tool_necessary=0 samples",
            },
            "summary_rows": summary_rows,
        },
    )
    print(f"saved stage-6 outputs: {output_root}")


if __name__ == "__main__":
    main()
