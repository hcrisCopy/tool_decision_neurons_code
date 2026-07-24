#!/usr/bin/env python3
"""Stage 9 CTD neuron training.

It follows the experiment plan and the Who Transfers Safety
neuron-aware training style: freeze normal parameters and update only the
target CTD neurons.

Neuron definition:
- Q/K/V neurons are rows of W_Q/W_K/W_V.
- O neurons are columns of W_O.

For each hop subset this script trains an independent CTD delta checkpoint from
the base model, then saves only the selected neuron before/after/delta tensors.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
STAGE6_DIR = CODE_ROOT / "05_single_type_causal_validation"
for path in (str(STAGE6_DIR), str(CODE_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import run_single_type_causal_validation as stage6  # noqa: E402
from common.io_utils import write_json, write_jsonl  # noqa: E402

TASK_TYPES = ("A", "B", "C")
SUBSETS = ("single_hop", "multi_hop")
MATRICES = ("Q", "K", "V", "O")


@dataclass
class SFTExample:
    sample_uid: str
    subset: str
    task_type: str
    env_name: str
    tool_necessary: int
    messages: List[Dict[str, str]]
    tools: List[Dict[str, Any]]
    assistant_messages: int
    tool_calls: int
    trajectory_mode: str


class TokenizedSFTDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-alias", default="qwen3-4b-instruct")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--modified-dir", required=True)
    parser.add_argument("--shared-dir", required=True, help="Stage-7 shared_by_subset dir.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-types", nargs="+", default=list(TASK_TYPES), choices=TASK_TYPES)
    parser.add_argument("--subsets", nargs="+", default=list(SUBSETS), choices=SUBSETS)
    parser.add_argument("--split", default="train")
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
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def format_tool_call(tool_name: str, arguments: Dict[str, Any], tool_format: str) -> str:
    body = json.dumps({"name": tool_name, "arguments": arguments}, ensure_ascii=False)
    if tool_format == "native":
        return body
    return "<tool_call>\n" + body + "\n</tool_call>"


def tool_response_message(tool_result: Dict[str, Any]) -> Dict[str, str]:
    content = json.dumps(tool_result, ensure_ascii=False)
    return {"role": "user", "content": "<tool_response>\n" + content + "\n</tool_response>"}


def final_answer_message(answer: Any) -> Dict[str, str]:
    return {"role": "assistant", "content": "\\boxed{" + str(answer) + "}"}


def safe_parse_steps(task: Dict[str, Any]) -> List[str]:
    steps = (task.get("expected") or {}).get("steps") or []
    return [str(step) for step in steps]


MONTH_TO_NUM = {
    name: idx
    for idx, name in enumerate(
        [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
        start=1,
    )
}


def literal_eval_str(text: str) -> Any:
    return ast.literal_eval(text.strip())


def first_bracket_literal(text: str, start: int = 0) -> Tuple[Any, int, int]:
    begin = text.find("[", start)
    if begin < 0:
        raise ValueError("No bracket literal found.")
    depth = 0
    for pos in range(begin, len(text)):
        char = text[pos]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return literal_eval_str(text[begin : pos + 1]), begin, pos + 1
    raise ValueError("Unclosed bracket literal.")


def parse_key_value_args(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    matches = list(re.finditer(r"(axis|index|value)=", text))
    if not matches:
        return out
    for idx, match in enumerate(matches):
        key = match.group(1)
        value_start = match.end()
        value_end = matches[idx + 1].start() - 1 if idx + 1 < len(matches) else len(text)
        raw = text[value_start:value_end].strip().strip(",")
        if key in {"axis", "index"}:
            out[key] = int(raw)
        else:
            out[key] = literal_eval_str(raw)
    return out


def date_iso(month_name: str, day: str, year: str) -> str:
    month = MONTH_TO_NUM[month_name]
    return f"{int(year):04d}-{month:02d}-{int(day):02d}"


def extract_code_blocks(text: str) -> List[str]:
    return [match.group(1).strip() for match in re.finditer(r"```(?:python)?\s*([\s\S]*?)```", text)]


def calculator_calls(task: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any], str]]:
    instruction = str(task["instruction"])
    if "Three-step computation:" in instruction:
        match = re.search(
            r"First compute x = (?P<x>.*?)\. Then compute y = (?P<y>.*?)\. Finally compute z = (?P<z>.*?)\.",
            instruction,
            flags=re.S,
        )
        if not match:
            raise ValueError("Cannot parse three-step calculator instruction.")
        values: Dict[str, str] = {}
        calls: List[Tuple[str, Dict[str, Any], str]] = []
        for name in ("x", "y", "z"):
            expr = match.group(name).strip()
            for variable, value in values.items():
                expr = re.sub(rf"\b{re.escape(variable)}\b", value, expr)
            calls.append(("evaluate_expression", {"expression": expr}, "parsed_calculator_step"))
            # Use Python's arithmetic only to substitute later expressions; the
            # actual training trajectory still calls the CalculatorEnv tool.
            values[name] = str(eval(expr, {"__builtins__": {}}, {}))
        return calls
    match = re.search(r"(?:number:|number\s*:)\s*(?P<expr>.+)$", instruction, flags=re.S)
    if not match:
        match = re.search(r"Compute exactly and return only number:\s*(?P<expr>.+)$", instruction, flags=re.S)
    if not match:
        raise ValueError("Cannot parse calculator expression.")
    expr = match.group("expr").strip().rstrip(".")
    return [("evaluate_expression", {"expression": expr}, "parsed_calculator_expression")]


def decoding_calls(task: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any], str]]:
    instruction = str(task["instruction"])
    patterns = [
        (
            r"Encode '([^']+)' in Morse code",
            lambda m: ("encode", {"message": m.group(1), "encoding_type": "morse"}, "parsed_morse_encode"),
        ),
        (
            r"Decode this Morse code: '([^']+)'",
            lambda m: ("decode", {"encoded_message": m.group(1), "encoding_type": "morse"}, "parsed_morse_decode"),
        ),
        (
            r"Apply Caesar cipher with shift (\d+) to '([^']+)'",
            lambda m: ("encode", {"message": m.group(2), "encoding_type": "caesar", "key": int(m.group(1))}, "parsed_caesar_encode"),
        ),
        (
            r"Encode '([^']+)' using Caesar cipher with shift (\d+)",
            lambda m: ("encode", {"message": m.group(1), "encoding_type": "caesar", "key": int(m.group(2))}, "parsed_caesar_encode"),
        ),
        (
            r"Decode '([^']+)' using Caesar cipher with shift (\d+)",
            lambda m: ("decode", {"encoded_message": m.group(1), "encoding_type": "caesar", "key": int(m.group(2))}, "parsed_caesar_decode"),
        ),
        (
            r"Encode '([^']+)' using the custom cipher called '([^']+)'",
            lambda m: ("encode", {"message": m.group(1), "encoding_type": "custom", "key": m.group(2)}, "parsed_custom_encode"),
        ),
        (
            r"Decode '([^']+)' using the custom cipher called '([^']+)'",
            lambda m: ("decode", {"encoded_message": m.group(1), "encoding_type": "custom", "key": m.group(2)}, "parsed_custom_decode"),
        ),
    ]
    for pattern, build in patterns:
        match = re.search(pattern, instruction)
        if match:
            return [build(match)]
    raise ValueError("Cannot parse decoding instruction.")


def code_executor_calls(task: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any], str]]:
    blocks = extract_code_blocks(str(task["instruction"]))
    if not blocks:
        raise ValueError("Cannot parse code block.")
    calls = [("run_code", {"code": blocks[0]}, "parsed_code_block")]
    steps = safe_parse_steps(task)
    # Deterministic fallback for later multi-hop steps: the dataset stores expected
    # intermediate values but not executable step-2/3 code. We keep the first
    # real code block and replay remaining intermediate values through run_code
    # so the trajectory still contains tool calls and tool responses.
    for value in steps[1:]:
        calls.append(("run_code", {"code": "print(" + json.dumps(value, ensure_ascii=False) + ")"}, "replay_intermediate_value"))
    return calls


def retriever_calls(task: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any], str]]:
    instruction = str(task["instruction"])
    queries = re.findall(r"Step\s+\d+:\s*(.*?)\s*Call the answer", instruction, flags=re.S)
    calls: List[Tuple[str, Dict[str, Any], str]] = []
    envs, _tools = stage6.build_envs(task)
    if not queries:
        expected_answer = str((task.get("expected") or {}).get("answer", "")).strip().lower()
        env_cfg = (task.get("environments") or [{}])[0]
        corpus = (env_cfg.get("parameters") or {}).get("corpus") or []
        matched_doc = None
        for doc in corpus:
            content = f"{doc.get('title', '')} {doc.get('text', '')}".lower()
            if expected_answer and expected_answer in content:
                matched_doc = doc
                break
        if matched_doc is None and corpus:
            matched_doc = corpus[0]
        if matched_doc is None:
            raise ValueError("No retriever corpus document available.")
        query = str(matched_doc.get("title") or expected_answer or task["instruction"])
        queries = [query]

    for query in queries:
        query = re.sub(r"\s+", " ", query).strip()
        search_result = stage6.route_tool_call(envs, "search_corpus", {"query": query, "top_k": 1})
        calls.append(("search_corpus", {"query": query, "top_k": 1}, "parsed_retriever_search"))
        hit_id = ""
        try:
            hits = search_result.get("data", {}).get("hits", [])
            if hits:
                hit_id = hits[0].get("id", "")
        except Exception:
            hit_id = ""
        if hit_id:
            calls.append(("read_doc", {"doc_id": hit_id}, "parsed_retriever_read_doc"))
    if not calls:
        raise ValueError("No retriever calls built.")
    return calls


def counting_calls(task: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any], str]]:
    instruction = str(task["instruction"])
    match = re.search(r"C\((\d+)\s*,\s*(\d+)\)", instruction)
    if match:
        return [("combination", {"n": int(match.group(1)), "k": int(match.group(2))}, "parsed_counting_combination")]
    match = re.search(r"choose\s+(\d+)\s+items\s+from\s+(\d+)", instruction)
    if match:
        return [("combination", {"n": int(match.group(2)), "k": int(match.group(1))}, "parsed_counting_combination")]
    match = re.search(r"P\((\d+)\s*,\s*(\d+)\)", instruction)
    if match:
        return [("permutation", {"n": int(match.group(1)), "k": int(match.group(2))}, "parsed_counting_permutation")]
    match = re.search(r"arrange\s+(\d+)\s+items\s+from\s+(\d+)", instruction)
    if match:
        return [("permutation", {"n": int(match.group(2)), "k": int(match.group(1))}, "parsed_counting_permutation")]
    match = re.search(r"What is\s+(\d+)!", instruction)
    if match:
        return [("factorial", {"n": int(match.group(1))}, "parsed_counting_factorial")]
    raise ValueError("Cannot parse counting instruction.")


def statistics_calls(task: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any], str]]:
    instruction = str(task["instruction"])
    if "Pearson correlation coefficient" in instruction:
        match = re.search(r"X=(\[.*?\])\s+and\s+Y=(\[.*?\])\?", instruction, flags=re.S)
        if not match:
            raise ValueError("Cannot parse correlation lists.")
        return [
            (
                "compute_stat",
                {
                    "data": literal_eval_str(match.group(1)),
                    "data2": literal_eval_str(match.group(2)),
                    "stat_type": "correlation",
                },
                "parsed_statistics_correlation",
            )
        ]
    data, _begin, _end = first_bracket_literal(instruction)
    lowered = instruction.lower()
    if "range (max minus min)" in lowered:
        return [("describe", {"data": data}, "parsed_statistics_describe_for_range")]
    stat_type = None
    for name, value in [
        ("standard deviation", "std"),
        ("population variance", "variance"),
        ("variance", "variance"),
        ("median", "median"),
        ("mean", "mean"),
        ("sum", "sum"),
    ]:
        if name in lowered:
            stat_type = value
            break
    if stat_type is None:
        raise ValueError("Cannot parse statistic type.")
    return [("compute_stat", {"data": data, "stat_type": stat_type}, f"parsed_statistics_{stat_type}")]


def matrix_calls(task: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any], str]]:
    instruction = str(task["instruction"])
    matrix, _begin, _end = first_bracket_literal(instruction)
    lowered = instruction.lower()
    if "trace" in lowered:
        return [("matrix_trace", {"matrix": matrix}, "parsed_matrix_trace")]
    if "determinant" in lowered:
        return [("matrix_determinant", {"matrix": matrix}, "parsed_matrix_determinant")]
    raise ValueError("Cannot parse matrix instruction.")


def prime_calls(task: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any], str]]:
    instruction = str(task["instruction"])
    match = re.search(r"Is\s+(\d+)\s+(?:a\s+)?prime", instruction)
    if match:
        return [("is_prime", {"n": int(match.group(1))}, "parsed_prime_is_prime")]
    match = re.search(r"What is the\s+(\d+)(?:st|nd|rd|th)\s+prime", instruction)
    if match:
        return [("nth_prime", {"n": int(match.group(1))}, "parsed_prime_nth_prime")]
    match = re.search(r"prime factorization of\s+(\d+)", instruction)
    if match:
        return [("factorize", {"n": int(match.group(1))}, "parsed_prime_factorize")]
    raise ValueError("Cannot parse prime instruction.")


def hash_calls(task: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any], str]]:
    instruction = str(task["instruction"])
    match = re.search(r"What is the\s+([A-Za-z0-9_]+)\s+hash of\s+(?:the empty string ''|'([^']*)')", instruction)
    if not match:
        raise ValueError("Cannot parse hash instruction.")
    algorithm = match.group(1).lower()
    input_string = match.group(2) if match.group(2) is not None else ""
    return [("compute_hash", {"input_string": input_string, "algorithm": algorithm}, "parsed_hash_compute")]


def datetime_calls(task: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any], str]]:
    instruction = str(task["instruction"])
    match = re.search(r"How many days are between (\w+) (\d+), (\d+) and (\w+) (\d+), (\d+)", instruction)
    if match:
        return [
            (
                "date_diff",
                {
                    "date1": date_iso(match.group(1), match.group(2), match.group(3)),
                    "date2": date_iso(match.group(4), match.group(5), match.group(6)),
                },
                "parsed_datetime_diff",
            )
        ]
    match = re.search(r"How many days are between (\w+) (\d+) and (\w+) (\d+), (\d+)", instruction)
    if match:
        return [
            (
                "date_diff",
                {
                    "date1": date_iso(match.group(1), match.group(2), match.group(5)),
                    "date2": date_iso(match.group(3), match.group(4), match.group(5)),
                },
                "parsed_datetime_diff",
            )
        ]
    match = re.search(r"What date is (\d+) days after (\w+) (\d+), (\d+)", instruction)
    if match:
        return [
            (
                "date_add",
                {
                    "date": date_iso(match.group(2), match.group(3), match.group(4)),
                    "days": int(match.group(1)),
                },
                "parsed_datetime_add",
            )
        ]
    match = re.search(r"What day of the week is (\w+) (\d+), (\d+)", instruction)
    if match:
        return [("day_of_week", {"date": date_iso(match.group(1), match.group(2), match.group(3))}, "parsed_datetime_day_of_week")]
    raise ValueError("Cannot parse datetime instruction.")


def list_calls(task: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any], str]]:
    instruction = str(task["instruction"])
    init_marker = "Initial "
    apply_marker = ". Apply "
    start = instruction.find(init_marker)
    middle = instruction.find(apply_marker)
    if start < 0 or middle < 0:
        raise ValueError("Cannot parse list instruction.")
    values = literal_eval_str(instruction[start + len(init_marker) : middle])
    op_text = instruction[middle + len(apply_marker) :].split(". Return", 1)[0].strip()
    match = re.match(r"(\w+)\((.*)\)$", op_text)
    if not match:
        raise ValueError("Cannot parse list operation.")
    tool_name = match.group(1)
    arg_text = match.group(2).strip()
    arguments: Dict[str, Any] = {"values": values}
    if arg_text:
        if "=" in arg_text:
            arguments.update(parse_key_value_args(arg_text))
        elif tool_name == "remove":
            arguments["value"] = literal_eval_str(arg_text)
    return [(tool_name, arguments, f"parsed_list_{tool_name}")]


def schedule_calls(task: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any], str]]:
    instruction = str(task["instruction"])
    meetings, _begin, _end = first_bracket_literal(instruction)
    lowered = instruction.lower()
    if "overlap" in lowered:
        return [("check_conflict", {"intervals": meetings}, "parsed_schedule_conflict")]
    times = re.search(r"between\s+(\d{2}:\d{2})\s+and\s+(\d{2}:\d{2})", instruction)
    if not times:
        raise ValueError("Cannot parse schedule time window.")
    if "1-hour" in lowered:
        min_duration = 60
    else:
        duration_match = re.search(r"(\d+)-minute", lowered)
        if not duration_match:
            raise ValueError("Cannot parse schedule duration.")
        min_duration = int(duration_match.group(1))
    return [
        (
            "find_free_slots",
            {
                "busy": meetings,
                "day_start": times.group(1),
                "day_end": times.group(2),
                "min_duration": min_duration,
            },
            "parsed_schedule_free_slots",
        )
    ]


def regex_calls(task: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any], str]]:
    instruction = str(task["instruction"])
    match = re.search(r"r'(.+?)'", instruction, flags=re.S)
    if not match:
        raise ValueError("Cannot parse regex pattern.")
    pattern = match.group(1)
    text_match = re.search(r'"(.*?)"', instruction[match.end() :], flags=re.S)
    if not text_match:
        raise ValueError("Cannot parse regex text.")
    operation = "search" if "search" in instruction.lower() else "findall"
    return [("regex_match", {"pattern": pattern, "text": text_match.group(1), "operation": operation}, f"parsed_regex_{operation}")]


def corpus_lookup_calls(task: Dict[str, Any], tool_name: str, arg_name: str, mode: str) -> List[Tuple[str, Dict[str, Any], str]]:
    instruction = str(task["instruction"])
    if tool_name == "lookup_year":
        match = re.search(r"In what year was (.*?)\?", instruction)
        if match:
            return [(tool_name, {arg_name: match.group(1).strip()}, mode)]
    query = instruction.split(" Answer ", 1)[0].strip()
    return [(tool_name, {arg_name: query}, mode)]


def build_tool_calls(task: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any], str]]:
    env_name = str(task.get("env_name"))
    if env_name == "CalculatorEnv":
        return calculator_calls(task)
    if env_name == "CountingEnv":
        return counting_calls(task)
    if env_name == "StatisticsEnv":
        return statistics_calls(task)
    if env_name == "MatrixEnv":
        return matrix_calls(task)
    if env_name == "PrimeEnv":
        return prime_calls(task)
    if env_name == "HashEnv":
        return hash_calls(task)
    if env_name == "DateTimeEnv":
        return datetime_calls(task)
    if env_name == "ListManipulationEnv":
        return list_calls(task)
    if env_name == "ScheduleEnv":
        return schedule_calls(task)
    if env_name == "RegexMatchEnv":
        return regex_calls(task)
    if env_name == "HistoricalYearEnv":
        return corpus_lookup_calls(task, "lookup_year", "event", "parsed_historical_lookup")
    if env_name == "GameRuleEnv":
        return corpus_lookup_calls(task, "lookup_rule", "query", "parsed_game_rule_lookup")
    if env_name == "DecodingEnv":
        return decoding_calls(task)
    if env_name == "CodeExecutorEnv":
        return code_executor_calls(task)
    if env_name == "RetrieverEnv":
        return retriever_calls(task)
    raise ValueError(f"Unsupported env for deterministic training trajectory: {env_name}")


def make_sft_example(task: Dict[str, Any], tool_format: str) -> Tuple[Optional[SFTExample], Optional[str]]:
    system_prompt = stage6.system_prompt_for(tool_format)
    envs, tools = stage6.build_envs(task)
    messages = [{"role": "system", "content": system_prompt}]
    if stage6.has_environment(task, "ListManipulationEnv"):
        messages.append({"role": "system", "content": stage6.LIST_MANIPULATION_FORMAT_CONTRACT})
    messages.append({"role": "user", "content": stage6.build_current_no_reasoning_user_message(task["instruction"])})
    answer = (task.get("expected") or {}).get("answer", "")
    if int(task.get("tool_necessary", 0)) == 0:
        messages.append(final_answer_message(answer))
        return (
            SFTExample(
                sample_uid=str(task.get("sample_uid")),
                subset=str(task.get("subset")),
                task_type=str(task.get("task_type")),
                env_name=str(task.get("env_name")),
                tool_necessary=0,
                messages=messages,
                tools=tools,
                assistant_messages=1,
                tool_calls=0,
                trajectory_mode="direct_no_tool",
            ),
            None,
        )

    try:
        calls = build_tool_calls(task)
    except Exception as exc:
        return None, f"trajectory_build_failed: {type(exc).__name__}: {str(exc)[:200]}"

    modes: List[str] = []
    tool_calls = 0
    for tool_name, arguments, mode in calls:
        messages.append({"role": "assistant", "content": format_tool_call(tool_name, arguments, tool_format)})
        result = stage6.route_tool_call(envs, tool_name, deepcopy(arguments))
        messages.append(tool_response_message(result))
        tool_calls += 1
        modes.append(mode)
    messages.append(final_answer_message(answer))
    return (
        SFTExample(
            sample_uid=str(task.get("sample_uid")),
            subset=str(task.get("subset")),
            task_type=str(task.get("task_type")),
            env_name=str(task.get("env_name")),
            tool_necessary=1,
            messages=messages,
            tools=tools,
            assistant_messages=tool_calls + 1,
            tool_calls=tool_calls,
            trajectory_mode="+".join(sorted(set(modes))),
        ),
        None,
    )


def render_chat(tokenizer: Any, messages: List[Dict[str, str]], tools: List[Dict[str, Any]], enable_thinking: Optional[bool], add_generation_prompt: bool) -> str:
    kwargs: Dict[str, Any] = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
    if tools:
        kwargs["tools"] = tools
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def assistant_spans(tokenizer: Any, messages: List[Dict[str, str]], tools: List[Dict[str, Any]], enable_thinking: Optional[bool]) -> Tuple[str, List[Tuple[int, int]]]:
    full_text = render_chat(tokenizer, messages, tools, enable_thinking, add_generation_prompt=False)
    spans: List[Tuple[int, int]] = []
    for idx, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        prefix = render_chat(tokenizer, messages[:idx], tools, enable_thinking, add_generation_prompt=True)
        upto = render_chat(tokenizer, messages[: idx + 1], tools, enable_thinking, add_generation_prompt=False)
        start = len(prefix)
        end = len(upto)
        if start < end <= len(full_text):
            spans.append((start, end))
        else:
            content = str(message.get("content", ""))
            pos = full_text.find(content)
            if pos >= 0:
                spans.append((pos, pos + len(content)))
    return full_text, spans


def tokenize_example(tokenizer: Any, example: SFTExample, enable_thinking: Optional[bool], max_length: int) -> Dict[str, Any]:
    full_text, spans = assistant_spans(tokenizer, example.messages, example.tools, enable_thinking)
    try:
        tokenized = tokenizer(
            full_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_length,
        )
    except (NotImplementedError, TypeError):
        tokenized = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )
    input_ids = list(tokenized["input_ids"])
    attention_mask = list(tokenized["attention_mask"])
    offsets = tokenized.get("offset_mapping") or []
    labels = [-100] * len(input_ids)
    supervised_tokens = 0
    if offsets:
        for i, (start, end) in enumerate(offsets):
            if end <= start:
                continue
            if any(end > span_start and start < span_end for span_start, span_end in spans):
                labels[i] = input_ids[i]
                supervised_tokens += 1
    else:
        for span_start, span_end in spans:
            prefix_len = len(tokenizer(full_text[:span_start], add_special_tokens=False)["input_ids"])
            span_len = len(tokenizer(full_text[span_start:span_end], add_special_tokens=False)["input_ids"])
            for i in range(prefix_len, min(prefix_len + span_len, len(labels))):
                labels[i] = input_ids[i]
                supervised_tokens += 1
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "sample_uid": example.sample_uid,
        "subset": example.subset,
        "task_type": example.task_type,
        "env_name": example.env_name,
        "tool_necessary": example.tool_necessary,
        "assistant_messages": example.assistant_messages,
        "tool_calls": example.tool_calls,
        "trajectory_mode": example.trajectory_mode,
        "token_count": len(input_ids),
        "supervised_tokens": supervised_tokens,
    }


def collate_batch(tokenizer: Any, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    max_len = max(len(row["input_ids"]) for row in batch)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids, attention_mask, labels = [], [], []
    for row in batch:
        pad_len = max_len - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [pad_id] * pad_len)
        attention_mask.append(row["attention_mask"] + [0] * pad_len)
        labels.append(row["labels"] + [-100] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def load_train_tasks(modified_dir: Path, subset: str, split: str, task_types: Iterable[str]) -> List[Dict[str, Any]]:
    grouped = stage6.load_test_tasks(modified_dir, [subset], split, task_types, 0)
    rows: List[Dict[str, Any]] = []
    for task_type in task_types:
        rows.extend(grouped[task_type])
    rows.sort(key=lambda row: (str(row.get("task_type")), int(row.get("id") or 0)))
    return rows


def load_ctd_rows(shared_dir: Path, subset: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    subset_dir = shared_dir / subset
    rows = read_jsonl(subset_dir / "CTD_neurons.jsonl")
    manifest = read_json(subset_dir / "manifest.json")
    return rows, manifest


def torch_dtype(name: str) -> Any:
    return stage6.torch_dtype(name)


def enable_thinking_value(value: str) -> Optional[bool]:
    return stage6.enable_thinking_value(value)


def layer_modules(model: Any) -> List[Any]:
    return stage6.layer_modules(model)


def module_for_neuron(model: Any, layer: int, matrix: str) -> Tuple[str, torch.nn.Module, str]:
    attn = layer_modules(model)[layer].self_attn
    if matrix == "Q":
        return f"model.layers.{layer}.self_attn.q_proj.weight", attn.q_proj, "row"
    if matrix == "K":
        return f"model.layers.{layer}.self_attn.k_proj.weight", attn.k_proj, "row"
    if matrix == "V":
        return f"model.layers.{layer}.self_attn.v_proj.weight", attn.v_proj, "row"
    if matrix == "O":
        return f"model.layers.{layer}.self_attn.o_proj.weight", attn.o_proj, "column"
    raise ValueError(f"Unsupported matrix: {matrix}")


def build_gradient_masks(model: Any, ctd_rows: List[Dict[str, Any]]) -> Tuple[Dict[torch.nn.Parameter, torch.Tensor], List[Dict[str, Any]]]:
    masks: Dict[torch.nn.Parameter, torch.Tensor] = {}
    trainable_rows: List[Dict[str, Any]] = []
    for row in ctd_rows:
        layer = int(row["layer"])
        matrix = str(row["matrix"])
        index = int(row["index"])
        param_name, module, orientation = module_for_neuron(model, layer, matrix)
        weight = module.weight
        mask = masks.get(weight)
        if mask is None:
            mask = torch.zeros_like(weight, dtype=torch.float32, device=weight.device)
            masks[weight] = mask
        if orientation == "row":
            if index >= weight.shape[0]:
                raise IndexError(f"{param_name} row index out of range: {index}")
            mask[index, :] = 1.0
            trainable_count = int(weight.shape[1])
        else:
            if index >= weight.shape[1]:
                raise IndexError(f"{param_name} column index out of range: {index}")
            mask[:, index] = 1.0
            trainable_count = int(weight.shape[0])
        trainable_rows.append(
            {
                "layer": layer,
                "matrix": matrix,
                "index": index,
                "param_name": param_name,
                "orientation": orientation,
                "trainable_scalar_count": trainable_count,
                "neuron_id": row.get("neuron_id", f"L{layer:02d}.{matrix}.{index:05d}"),
            }
        )

        bias = getattr(module, "bias", None)
        if orientation == "row" and bias is not None:
            bias_mask = masks.get(bias)
            if bias_mask is None:
                bias_mask = torch.zeros_like(bias, dtype=torch.float32, device=bias.device)
                masks[bias] = bias_mask
            bias_mask[index] = 1.0
    return masks, trainable_rows


def freeze_and_hook(model: Any, masks: Dict[torch.nn.Parameter, torch.Tensor]) -> Tuple[List[Any], List[torch.nn.Parameter]]:
    for param in model.parameters():
        param.requires_grad_(False)
    hooks = []
    params: List[torch.nn.Parameter] = []
    for param, mask in masks.items():
        param.requires_grad_(True)
        params.append(param)
        hooks.append(param.register_hook(lambda grad, mask=mask: grad * mask.to(dtype=grad.dtype, device=grad.device)))
    return hooks, params


def selected_param_snapshots(masks: Dict[torch.nn.Parameter, torch.Tensor]) -> Dict[torch.nn.Parameter, torch.Tensor]:
    return {param: param.detach().float().cpu().clone() for param in masks}


def selected_neuron_deltas(
    ctd_rows: List[Dict[str, Any]],
    model: Any,
    before: Dict[torch.nn.Parameter, torch.Tensor],
    masks: Dict[torch.nn.Parameter, torch.Tensor],
    include_full_snapshot: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    deltas: List[Dict[str, Any]] = []
    max_masked_change = 0.0
    max_unmasked_change = 0.0
    for row in ctd_rows:
        layer = int(row["layer"])
        matrix = str(row["matrix"])
        index = int(row["index"])
        param_name, module, orientation = module_for_neuron(model, layer, matrix)
        weight = module.weight
        before_weight = before[weight]
        after_weight = weight.detach().float().cpu()
        mask = masks[weight].detach().float().cpu()
        change = after_weight - before_weight
        if mask.numel():
            max_masked_change = max(max_masked_change, float((change * mask).abs().max().item()))
            max_unmasked_change = max(max_unmasked_change, float((change * (1.0 - mask)).abs().max().item()))
        if orientation == "row":
            before_slice = before_weight[index, :].clone()
            after_slice = after_weight[index, :].clone()
        else:
            before_slice = before_weight[:, index].clone()
            after_slice = after_weight[:, index].clone()
        item: Dict[str, Any] = {
            "layer": layer,
            "matrix": matrix,
            "index": index,
            "neuron_id": row.get("neuron_id", f"L{layer:02d}.{matrix}.{index:05d}"),
            "param_name": param_name,
            "orientation": orientation,
            "before": before_slice,
            "after": after_slice,
            "delta": after_slice - before_slice,
        }
        if include_full_snapshot:
            item["full_param_before"] = before_weight
            item["full_param_after"] = after_weight
            item["full_param_mask"] = mask
        deltas.append(item)
    return deltas, {"max_masked_change": max_masked_change, "max_unmasked_change": max_unmasked_change}


def prepare_examples(args: argparse.Namespace, tokenizer: Any, subset: str, tool_format: str, enable_thinking: Optional[bool]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    tasks = load_train_tasks(Path(args.modified_dir), subset, args.split, args.task_types)
    examples: List[SFTExample] = []
    skipped: List[Dict[str, Any]] = []
    for task in tasks:
        example, error = make_sft_example(task, tool_format)
        if example is None:
            skipped.append(
                {
                    "sample_uid": task.get("sample_uid"),
                    "subset": task.get("subset"),
                    "task_type": task.get("task_type"),
                    "env_name": task.get("env_name"),
                    "tool_necessary": task.get("tool_necessary"),
                    "error": error,
                }
            )
        else:
            examples.append(example)
    tokenized = [tokenize_example(tokenizer, example, enable_thinking, args.max_length) for example in examples]
    tokenized = [row for row in tokenized if int(row["supervised_tokens"]) > 0]
    metadata = [
        {
            "sample_uid": example.sample_uid,
            "subset": example.subset,
            "task_type": example.task_type,
            "env_name": example.env_name,
            "tool_necessary": example.tool_necessary,
            "assistant_messages": example.assistant_messages,
            "tool_calls": example.tool_calls,
            "trajectory_mode": example.trajectory_mode,
        }
        for example in examples
    ]
    return tokenized, metadata, skipped


def train_subset(args: argparse.Namespace, subset: str) -> Dict[str, Any]:
    output_root = Path(args.output_dir) / args.model_alias / "neuron_training_by_subset" / subset
    if output_root.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_root}. Use --overwrite.")
    output_root.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    enable_thinking = enable_thinking_value(args.enable_thinking)
    tool_format = stage6.detect_tool_format(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    ctd_rows, shared_manifest = load_ctd_rows(Path(args.shared_dir), subset)
    tokenized, metadata, skipped = prepare_examples(args, tokenizer, subset, tool_format, enable_thinking)
    if not tokenized:
        raise ValueError(f"No trainable SFT examples for subset {subset}.")
    write_jsonl(output_root / "training_examples.jsonl", metadata)
    write_jsonl(output_root / "skipped_examples.jsonl", skipped)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype(args.torch_dtype),
        device_map=args.device_map,
        local_files_only=True,
    )
    model.train()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    masks, trainable_rows = build_gradient_masks(model, ctd_rows)
    hooks, params = freeze_and_hook(model, masks)
    before = selected_param_snapshots(masks)

    dataset = TokenizedSFTDataset(tokenized)
    loader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(tokenizer, batch),
    )
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=0.0)
    updates_per_epoch = max(1, math.ceil(len(loader) / max(1, args.gradient_accumulation_steps)))
    total_update_steps = max(1, int(args.epochs) * updates_per_epoch)
    warmup_steps = int(total_update_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_update_steps)
    device = model.get_input_embeddings().weight.device

    log_rows: List[Dict[str, Any]] = []
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    update_step = 0
    for epoch in range(int(args.epochs)):
        for batch_idx, batch in enumerate(loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / max(1, args.gradient_accumulation_steps)
            loss.backward()
            global_step += 1
            local_step = batch_idx + 1
            should_step = (local_step % max(1, args.gradient_accumulation_steps) == 0) or (local_step == len(loader))
            if should_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(params, args.max_gradient_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update_step += 1
                log_rows.append(
                    {
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "update_step": update_step,
                        "loss": float(loss.detach().float().item() * max(1, args.gradient_accumulation_steps)),
                        "grad_norm": float(grad_norm.detach().float().item()) if torch.is_tensor(grad_norm) else float(grad_norm),
                        "lr": float(scheduler.get_last_lr()[0]),
                    }
                )

    for hook in hooks:
        hook.remove()

    deltas, change_stats = selected_neuron_deltas(ctd_rows, model, before, masks, args.save_full_selected_param_snapshot)
    checkpoint = {
        "stage": "stage9_ctd_neuron_training_delta_checkpoint",
        "model_alias": args.model_alias,
        "model_path": args.model_path,
        "subset": subset,
        "neuron_definition": "Q/K/V neurons are W_Q/W_K/W_V rows; O neurons are W_O columns.",
        "gradient_mask_definition": "Only CTD_m neuron row/column parameters have nonzero gradients; all other gradients are zeroed by hooks.",
        "ctd_rows": ctd_rows,
        "trainable_neuron_deltas": deltas,
    }
    torch.save(checkpoint, output_root / "ctd_neuron_delta.pt")
    write_csv(output_root / "training_log.csv", log_rows, ["epoch", "global_step", "update_step", "loss", "grad_norm", "lr"])
    write_json(
        output_root / "trainable_mask_summary.json",
        {
            "trainable_neurons": trainable_rows,
            "trainable_neuron_count": len(ctd_rows),
            "trainable_scalar_count": int(sum(row["trainable_scalar_count"] for row in trainable_rows)),
            **change_stats,
        },
    )

    tool_needed = sum(int(row["tool_necessary"]) for row in tokenized)
    no_tool = len(tokenized) - tool_needed
    manifest = {
        "stage": "stage9_ctd_neuron_training",
        "model_alias": args.model_alias,
        "model_path": args.model_path,
        "modified_dir": args.modified_dir,
        "shared_dir": str(Path(args.shared_dir) / subset),
        "output_dir": str(output_root),
        "subset": subset,
        "split": args.split,
        "task_types": list(args.task_types),
        "epochs": args.epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "optimizer": "AdamW(weight_decay=0)",
        "global_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
        "loss": "Autoregressive cross-entropy on assistant tokens only; system/user/tool_response labels are -100.",
        "neuron_definition": "Who Transfers Safety attention projection neurons: Q/K/V projection rows and O output-projection columns.",
        "gradient_mask": "M_CTD(theta_i)=1 only for CTD row/column parameters; all other gradients are zero.",
        "ctd_size": len(ctd_rows),
        "ctd_exact_size": int(shared_manifest.get("exact_CTD_size", len(ctd_rows))),
        "train_examples": len(tokenized),
        "skipped_examples": len(skipped),
        "tool_necessary_1": tool_needed,
        "tool_necessary_0": no_tool,
        "total_input_tokens": int(sum(row["token_count"] for row in tokenized)),
        "total_supervised_tokens": int(sum(row["supervised_tokens"] for row in tokenized)),
        "training_log_rows": len(log_rows),
        "checkpoint": str(output_root / "ctd_neuron_delta.pt"),
        "mask_change_stats": change_stats,
    }
    write_json(output_root / "manifest.json", manifest)
    (output_root / "summary.md").write_text(summary_markdown(manifest, log_rows), encoding="utf-8")
    return manifest


def summary_markdown(manifest: Dict[str, Any], log_rows: List[Dict[str, Any]]) -> str:
    first_loss = log_rows[0]["loss"] if log_rows else float("nan")
    last_loss = log_rows[-1]["loss"] if log_rows else float("nan")
    lines = [
        "# Stage 9 CTD Neuron Training",
        "",
        f"- Model: `{manifest['model_alias']}`",
        f"- Subset: `{manifest['subset']}`",
        f"- CTD size: {manifest['ctd_size']} (exact={manifest['ctd_exact_size']})",
        f"- Train examples: {manifest['train_examples']} (tool=1: {manifest['tool_necessary_1']}, tool=0: {manifest['tool_necessary_0']})",
        f"- Global batch size: {manifest['global_batch_size']}",
        f"- Epochs: {manifest['epochs']}",
        f"- LR: {manifest['learning_rate']}",
        f"- First logged loss: {first_loss:.6f}",
        f"- Last logged loss: {last_loss:.6f}",
        f"- Max masked change: {manifest['mask_change_stats']['max_masked_change']:.8f}",
        f"- Max unmasked change: {manifest['mask_change_stats']['max_unmasked_change']:.8f}",
        "",
        "Outputs:",
        "- `ctd_neuron_delta.pt`: selected CTD before/after/delta tensors",
        "- `training_log.csv`: optimizer-step loss log",
        "- `training_examples.jsonl`: supervised trajectory metadata",
        "- `skipped_examples.jsonl`: skipped trajectory-build failures",
        "- `trainable_mask_summary.json`: row/column mask summary",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir) / args.model_alias / "neuron_training_by_subset"
    if output_root.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_root}. Use --overwrite.")
    output_root.mkdir(parents=True, exist_ok=True)

    manifests = []
    for subset in args.subsets:
        manifests.append(train_subset(args, subset))

    write_json(
        output_root / "manifest.json",
        {
            "stage": "stage9_ctd_neuron_training",
            "model_alias": args.model_alias,
            "model_path": args.model_path,
            "subsets": list(args.subsets),
            "output_dir": str(output_root),
            "subset_manifests": manifests,
        },
    )
    print(f"saved stage-9 CTD neuron training: {output_root}")


if __name__ == "__main__":
    main()
