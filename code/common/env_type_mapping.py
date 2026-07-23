"""When2Tool environment to A/B/C task-type mapping."""

from __future__ import annotations

from typing import Dict, List


TASK_TYPES: Dict[str, Dict[str, str]] = {
    "A": {
        "task_type_name": "computational_scale",
        "when2tool_category": "Computational Scale",
    },
    "B": {
        "task_type_name": "knowledge_boundaries",
        "when2tool_category": "Knowledge Boundaries",
    },
    "C": {
        "task_type_name": "execution_reliability",
        "when2tool_category": "Execution Reliability",
    },
}


ENV_TO_TASK_TYPE: Dict[str, str] = {
    # A: Computational Scale.
    "CalculatorEnv": "A",
    "StatisticsEnv": "A",
    "CountingEnv": "A",
    "MatrixEnv": "A",
    "PrimeEnv": "A",
    # B: Knowledge Boundaries.
    "RetrieverEnv": "B",
    "HistoricalYearEnv": "B",
    "GameRuleEnv": "B",
    "HashEnv": "B",
    "DecodingEnv": "B",
    # C: Execution Reliability.
    "ListManipulationEnv": "C",
    "DateTimeEnv": "C",
    "CodeExecutorEnv": "C",
    "ScheduleEnv": "C",
    "RegexMatchEnv": "C",
}


ENV_GROUPS: Dict[str, List[str]] = {
    task_type: [
        env_name
        for env_name, mapped_task_type in ENV_TO_TASK_TYPE.items()
        if mapped_task_type == task_type
    ]
    for task_type in TASK_TYPES
}


def get_task_type(env_name: str) -> str:
    """Return A/B/C task type for a When2Tool env_name."""
    try:
        return ENV_TO_TASK_TYPE[env_name]
    except KeyError as exc:
        known = ", ".join(sorted(ENV_TO_TASK_TYPE))
        raise KeyError(f"Unknown env_name: {env_name}. Known envs: {known}") from exc


def get_task_type_meta(env_name: str) -> Dict[str, str]:
    """Return task-type metadata for a When2Tool env_name."""
    task_type = get_task_type(env_name)
    meta = TASK_TYPES[task_type]
    return {
        "task_type": task_type,
        "task_type_name": meta["task_type_name"],
        "when2tool_category": meta["when2tool_category"],
    }


def mapping_rows() -> List[Dict[str, str]]:
    """Return mapping as rows for JSON/CSV export."""
    rows: List[Dict[str, str]] = []
    for env_name in sorted(ENV_TO_TASK_TYPE):
        meta = get_task_type_meta(env_name)
        rows.append({"env_name": env_name, **meta})
    return rows
