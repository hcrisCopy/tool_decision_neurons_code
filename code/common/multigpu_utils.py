"""Helpers for single-node multi-GPU stage runners."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


@dataclass
class Job:
    name: str
    cmd: List[str]
    cuda_device: Optional[str] = None
    env: Optional[Dict[str, str]] = None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_devices(devices: str, num_gpus: int) -> List[str]:
    parsed = [item.strip() for item in str(devices).split(",") if item.strip()]
    if not parsed:
        parsed = [str(idx) for idx in range(num_gpus)]
    if num_gpus <= 0:
        raise ValueError("--num-gpus must be positive.")
    if len(parsed) < num_gpus:
        raise ValueError(f"Need at least {num_gpus} CUDA devices, got {parsed}")
    return parsed[:num_gpus]


def run_jobs(jobs: Sequence[Job], max_parallel: int) -> None:
    if max_parallel <= 0:
        raise ValueError("max_parallel must be positive.")
    pending = list(jobs)
    active: List[tuple[Job, subprocess.Popen[Any]]] = []
    root = repo_root()
    base_env = os.environ.copy()
    base_env.setdefault("TOKENIZERS_PARALLELISM", "false")
    base_env.setdefault("PYTHONUNBUFFERED", "1")

    while pending or active:
        while pending and len(active) < max_parallel:
            job = pending.pop(0)
            env = base_env.copy()
            if job.env:
                env.update(job.env)
            if job.cuda_device is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(job.cuda_device)
            print(f"[launch] {job.name} cuda={env.get('CUDA_VISIBLE_DEVICES', 'all')}", flush=True)
            print("         " + " ".join(job.cmd), flush=True)
            proc = subprocess.Popen(job.cmd, cwd=str(root), env=env)
            active.append((job, proc))

        time.sleep(1.0)
        still_active: List[tuple[Job, subprocess.Popen[Any]]] = []
        for job, proc in active:
            code = proc.poll()
            if code is None:
                still_active.append((job, proc))
                continue
            if code != 0:
                for _, other in still_active:
                    other.terminate()
                for _, other in active:
                    if other.poll() is None:
                        other.terminate()
                raise RuntimeError(f"Job failed with exit code {code}: {job.name}")
            print(f"[done] {job.name}", flush=True)
        active = still_active


def command(script: str, *args: Any) -> List[str]:
    return [sys.executable, script, *[str(arg) for arg in args]]


def add_kv(cmd: List[str], key: str, value: Any) -> None:
    cmd.extend([key, str(value)])


def add_list(cmd: List[str], key: str, values: Iterable[Any]) -> None:
    cmd.append(key)
    cmd.extend(str(value) for value in values)


def add_flag(cmd: List[str], key: str, enabled: bool) -> None:
    if enabled:
        cmd.append(key)


def complete(path: Path, files: Iterable[str]) -> bool:
    return path.exists() and all((path / name).exists() for name in files)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def copytree(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            print(f"[skip] existing output: {dst}", flush=True)
            return
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def remove_if_exists(path: Path) -> None:
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def run_refresh(cmd: List[str]) -> None:
    print("[refresh] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(repo_root()), check=True)
