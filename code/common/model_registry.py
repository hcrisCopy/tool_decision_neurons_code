"""Model alias and path resolution for the project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import yaml


DEFAULT_MODELS: Dict[str, Dict[str, str]] = {
    "qwen3-1.7b": {
        "repo_id": "Qwen/Qwen3-1.7B",
        "local_path": "../Qwen/Qwen3-1.7B",
        "enable_thinking": "false",
    },
    "qwen3-4b-instruct": {
        "repo_id": "Qwen/Qwen3-4B-Instruct-2507",
        "local_path": "../Qwen/Qwen3-4B-Instruct-2507",
        "enable_thinking": "auto",
    },
    "qwen3-14b": {
        "repo_id": "Qwen/Qwen3-14B",
        "local_path": "../Qwen/Qwen3-14B",
        "enable_thinking": "false",
    },
    "qwen3-32b": {
        "repo_id": "Qwen/Qwen3-32B",
        "local_path": "../Qwen/Qwen3-32B",
        "enable_thinking": "false",
    },
    "llama3.1-8b": {
        "repo_id": "meta-llama/Llama-3.1-8B-Instruct",
        "local_path": "../meta-llama/Llama-3.1-8B-Instruct",
        "enable_thinking": "auto",
    },
    "llama3.3-70b": {
        "repo_id": "meta-llama/Llama-3.3-70B-Instruct",
        "local_path": "../meta-llama/Llama-3.3-70B-Instruct",
        "enable_thinking": "auto",
    },
}


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    repo_id: str
    local_path: str
    resolved_path: str
    resolved_from_local: bool
    enable_thinking: str = "auto"


def normalize_model_alias(name: str) -> str:
    return str(name or "").strip().lower().replace("_", "-")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_model_config(config_path: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    if config_path is None:
        config_path = str(repo_root() / "configs" / "models.yaml")
    path = Path(config_path)
    if not path.exists():
        return DEFAULT_MODELS
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    models = payload.get("models") or {}
    return {normalize_model_alias(k): dict(v) for k, v in models.items()}


def list_model_aliases(config_path: Optional[str] = None) -> Iterable[str]:
    return load_model_config(config_path).keys()


def resolve_model_spec(
    alias_or_path: str,
    config_path: Optional[str] = None,
    prefer_local: bool = True,
) -> ModelSpec:
    models = load_model_config(config_path)
    key = normalize_model_alias(alias_or_path)
    if key not in models:
        return ModelSpec(
            alias=Path(alias_or_path).name,
            repo_id=alias_or_path,
            local_path=alias_or_path,
            resolved_path=alias_or_path,
            resolved_from_local=False,
            enable_thinking="auto",
        )

    cfg = models[key]
    repo_id = str(cfg["repo_id"])
    local_path = str(cfg.get("local_path") or repo_id)
    resolved = repo_id
    resolved_from_local = False
    if prefer_local:
        candidate = (repo_root() / local_path).resolve()
        if candidate.exists():
            resolved = str(candidate)
            resolved_from_local = True
    return ModelSpec(
        alias=key,
        repo_id=repo_id,
        local_path=local_path,
        resolved_path=resolved,
        resolved_from_local=resolved_from_local,
        enable_thinking=str(cfg.get("enable_thinking") or "auto"),
    )
