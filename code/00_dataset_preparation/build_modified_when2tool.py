#!/usr/bin/env python3
"""Build modified When2Tool data with A/B/C task-type metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.env_type_mapping import ENV_TO_TASK_TYPE, TASK_TYPES, get_task_type_meta, mapping_rows

SUBSETS = ("single_hop", "multi_hop")
SPLITS = ("train", "test")
ORIGINAL_COLUMNS = (
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
    parser.add_argument("--data-root", default="../tool_decision_neurons_data")
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-parquet", action="store_true")
    return parser.parse_args()


def to_builtin(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def ensure_can_write(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Use --overwrite to replace it.")


def read_split(raw_dir: Path, subset: str, split: str) -> pd.DataFrame:
    split_dir = raw_dir / subset
    files = sorted(split_dir.glob(f"{split}-*.parquet")) or sorted(split_dir.glob(f"{split}*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found for {subset}/{split} in {split_dir}")

    frames = [pd.read_parquet(path) for path in files]
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    missing = [column for column in ORIGINAL_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {subset}/{split}: {missing}")

    unknown_envs = sorted(set(df["env_name"]) - set(ENV_TO_TASK_TYPE))
    if unknown_envs:
        raise ValueError(f"Unknown env_name values in {subset}/{split}: {unknown_envs}")
    return df


def build_record(row: Dict[str, Any], subset: str, split: str) -> Dict[str, Any]:
    meta = get_task_type_meta(str(row["env_name"]))
    original_id = to_builtin(row["id"])
    record: Dict[str, Any] = {
        "source_dataset": "When2Tool",
        "subset": subset,
        "split": split,
        "sample_uid": f"{subset}:{split}:{original_id}",
        "original_id": original_id,
        "task_type": meta["task_type"],
        "task_type_name": meta["task_type_name"],
        "when2tool_category": meta["when2tool_category"],
    }
    for column in ORIGINAL_COLUMNS:
        record[column] = to_builtin(row[column])
    return record


def write_jsonl(records: Iterable[Dict[str, Any]], path: Path, overwrite: bool) -> None:
    ensure_can_write(path, overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_records(records: List[Dict[str, Any]], output_dir: Path, subset: str, split: str, overwrite: bool, write_parquet: bool) -> None:
    subset_dir = output_dir / subset
    write_jsonl(records, subset_dir / f"{split}.jsonl", overwrite)
    if write_parquet:
        parquet_path = subset_dir / f"{split}.parquet"
        ensure_can_write(parquet_path, overwrite)
        pd.DataFrame(records).to_parquet(parquet_path, index=False)


def count_rows(records: List[Dict[str, Any]], subset: str, split: str) -> List[Dict[str, Any]]:
    df = pd.DataFrame(records)
    counts = df.groupby(["task_type", "task_type_name", "env_name", "difficulty"], dropna=False).size().reset_index(name="n")
    rows: List[Dict[str, Any]] = []
    for row in counts.to_dict(orient="records"):
        rows.append({
            "subset": subset,
            "split": split,
            "task_type": row["task_type"],
            "task_type_name": row["task_type_name"],
            "env_name": row["env_name"],
            "difficulty": row["difficulty"],
            "n": int(row["n"]),
        })
    return rows


def write_side_files(output_dir: Path, raw_dir_arg: str, output_dir_arg: str, split_stats: List[Dict[str, Any]], summary_rows: List[Dict[str, Any]], overwrite: bool, write_parquet: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.csv"
    ensure_can_write(summary_path, overwrite)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    mapping_path = output_dir / "env_type_mapping.json"
    ensure_can_write(mapping_path, overwrite)
    mapping_payload = {"task_types": TASK_TYPES, "env_to_task_type": ENV_TO_TASK_TYPE, "rows": mapping_rows()}
    mapping_path.write_text(json.dumps(mapping_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest_path = output_dir / "manifest.json"
    ensure_can_write(manifest_path, overwrite)
    manifest = {
        "dataset": "modified_when2tool",
        "source_dataset": "When2Tool",
        "raw_dir": raw_dir_arg,
        "output_dir": output_dir_arg,
        "write_parquet": write_parquet,
        "records": split_stats,
        "original_columns": list(ORIGINAL_COLUMNS),
        "added_columns": [
            "source_dataset",
            "subset",
            "split",
            "sample_uid",
            "original_id",
            "task_type",
            "task_type_name",
            "when2tool_category",
        ],
        "task_types": TASK_TYPES,
        "env_to_task_type": ENV_TO_TASK_TYPE,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    readme_path = output_dir / "README.md"
    ensure_can_write(readme_path, overwrite)
    readme_path.write_text(
        "# Modified When2Tool Dataset\n\n"
        "Generated from raw When2Tool parquet files. Original fields are preserved, and A/B/C task-type metadata is added.\n\n"
        "No model-specific tool_necessary labels are created in this stage.\n",
        encoding="utf-8",
    )

    baidu_path = output_dir / "baidu_netdisk_info.md"
    if not baidu_path.exists():
        baidu_path.write_text(
            "# Baidu Netdisk Handoff\n\n"
            "resource_name:\nnetdisk_url:\nextract_code:\narchive_name:\n"
            "target_dir: tool_decision_neurons_data/datasets/modified_when2tool/\n"
            "expected_files: manifest.json, summary.csv, env_type_mapping.json, single_hop/, multi_hop/\n"
            "usage: modified When2Tool dataset for labeling, probing, causal validation, and training.\n",
            encoding="utf-8",
        )


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    raw_dir_arg = args.raw_dir or str(data_root / "datasets" / "raw_when2tool")
    output_dir_arg = args.output_dir or str(data_root / "datasets" / "modified_when2tool")
    raw_dir = Path(raw_dir_arg)
    output_dir = Path(output_dir_arg)
    write_parquet = not args.no_parquet

    split_stats: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for subset in SUBSETS:
        for split in SPLITS:
            df = read_split(raw_dir, subset, split)
            records = [build_record(row, subset, split) for row in df.to_dict(orient="records")]
            write_records(records, output_dir, subset, split, args.overwrite, write_parquet)
            split_stats.append({
                "subset": subset,
                "split": split,
                "num_records": len(records),
                "task_type_counts": pd.Series([r["task_type"] for r in records]).value_counts().sort_index().to_dict(),
            })
            summary_rows.extend(count_rows(records, subset, split))

    write_side_files(output_dir, raw_dir_arg, output_dir_arg, split_stats, summary_rows, args.overwrite, write_parquet)

    print(f"Wrote modified dataset to: {output_dir}")
    for stat in split_stats:
        print(f"{stat['subset']}/{stat['split']}: {stat['num_records']} records, task_type_counts={stat['task_type_counts']}")


if __name__ == "__main__":
    main()
