#!/usr/bin/env python3
"""Build model-specific modified When2Tool datasets from generated 0/1 labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.env_type_mapping import ENV_TO_TASK_TYPE, TASK_TYPES, get_task_type_meta, mapping_rows  # noqa: E402

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
LABEL_COLUMNS = (
    "model_alias",
    "model_answer_raw",
    "model_answer_boxed",
    "model_answer",
    "no_tool_correct",
    "tool_necessary",
    "rounds",
    "tool_calls_rejected",
    "prompt_mode",
    "reasoning_mode",
    "num_shards",
    "shard_index",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="../tool_decision_neurons_data")
    parser.add_argument("--raw-dir", default="", help="Default: <data-root>/datasets/raw_when2tool")
    parser.add_argument("--labels-root", default="", help="Default: <data-root>/labels")
    parser.add_argument("--output-dir", default="", help="Default: <data-root>/datasets/modified_when2tool")
    parser.add_argument("--model-aliases", nargs="+", default=["all"], help="Model aliases under labels/. Use 'all' to scan labels root.")
    parser.add_argument("--subsets", nargs="+", default=list(SUBSETS), choices=SUBSETS)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    parser.add_argument("--max-samples", type=int, default=0, help="Demo/debug only. 0 means all raw samples.")
    parser.add_argument("--allow-partial", action="store_true", help="Allow building from incomplete labels; unlabeled raw samples are skipped.")
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


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                yield json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc


def write_json(path: Path, payload: Dict[str, Any], overwrite: bool) -> None:
    ensure_can_write(path, overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(records: Iterable[Dict[str, Any]], path: Path, overwrite: bool) -> None:
    ensure_can_write(path, overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def sample_uid(subset: str, split: str, raw_id: Any) -> str:
    return f"{subset}:{split}:{to_builtin(raw_id)}"


def read_raw_split(raw_dir: Path, subset: str, split: str, max_samples: int = 0) -> pd.DataFrame:
    split_dir = raw_dir / subset
    files = sorted(split_dir.glob(f"{split}-*.parquet")) or sorted(split_dir.glob(f"{split}*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found for {subset}/{split} in {split_dir}")

    frames = [pd.read_parquet(path) for path in files]
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0].copy()
    missing = [column for column in ORIGINAL_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required raw columns in {subset}/{split}: {missing}")

    unknown_envs = sorted(set(df["env_name"]) - set(ENV_TO_TASK_TYPE))
    if unknown_envs:
        raise ValueError(f"Unknown env_name values in {subset}/{split}: {unknown_envs}")
    if max_samples > 0:
        df = df.head(max_samples).reset_index(drop=True)
    return df


def label_files_for_split(labels_root: Path, model_alias: str, subset: str, split: str) -> List[Path]:
    split_dir = labels_root / model_alias / subset / split
    files: List[Path] = []
    direct = split_dir / "labels.jsonl"
    if direct.exists():
        files.append(direct)
    files.extend(sorted(split_dir.glob("shard_*_of_*/labels.jsonl")))
    if not files:
        raise FileNotFoundError(f"No labels.jsonl found for {model_alias}/{subset}/{split}: {split_dir}")
    return files


def read_label_split(labels_root: Path, model_alias: str, subset: str, split: str) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    labels: Dict[str, Dict[str, Any]] = {}
    source_files: List[str] = []
    for path in label_files_for_split(labels_root, model_alias, subset, split):
        source_files.append(str(path))
        for row in read_jsonl(path):
            uid = str(row.get("sample_uid") or sample_uid(subset, split, row.get("id")))
            if uid in labels:
                raise ValueError(f"Duplicate label for sample_uid={uid} while reading {model_alias}/{subset}/{split}")
            labels[uid] = row
    return labels, source_files


def validate_label(raw_row: Dict[str, Any], label: Dict[str, Any], model_alias: str, subset: str, split: str) -> None:
    uid = sample_uid(subset, split, raw_row["id"])
    if str(label.get("sample_uid") or uid) != uid:
        raise ValueError(f"Label sample_uid mismatch for {uid}: {label.get('sample_uid')}")
    if label.get("env_name") and str(label["env_name"]) != str(raw_row["env_name"]):
        raise ValueError(f"Label env_name mismatch for {uid}: {label.get('env_name')} != {raw_row['env_name']}")
    if label.get("task_type") and str(label["task_type"]) != get_task_type_meta(str(raw_row["env_name"]))["task_type"]:
        raise ValueError(f"Label task_type mismatch for {uid}")
    if label.get("model_alias") and str(label["model_alias"]) != model_alias:
        raise ValueError(f"Label model_alias mismatch for {uid}: {label.get('model_alias')} != {model_alias}")
    if "tool_necessary" not in label:
        raise ValueError(f"Missing tool_necessary in label for {uid}")


def build_record(raw_row: Dict[str, Any], label: Dict[str, Any], model_alias: str, subset: str, split: str) -> Dict[str, Any]:
    validate_label(raw_row, label, model_alias, subset, split)
    env_name = str(raw_row["env_name"])
    meta = get_task_type_meta(env_name)
    original_id = to_builtin(raw_row["id"])
    record: Dict[str, Any] = {
        "source_dataset": "When2Tool",
        "subset": subset,
        "split": split,
        "sample_uid": sample_uid(subset, split, original_id),
        "original_id": original_id,
        "task_type": meta["task_type"],
        "task_type_name": meta["task_type_name"],
        "when2tool_category": meta["when2tool_category"],
        "model_alias": model_alias,
    }
    for column in ORIGINAL_COLUMNS:
        record[column] = to_builtin(raw_row[column])
    for column in LABEL_COLUMNS:
        if column == "model_alias":
            continue
        record[column] = to_builtin(label.get(column))
    record["no_tool_correct"] = int(record["no_tool_correct"]) if record["no_tool_correct"] is not None else None
    record["tool_necessary"] = int(record["tool_necessary"])
    return record


def build_records_for_split(
    raw_dir: Path,
    labels_root: Path,
    model_alias: str,
    subset: str,
    split: str,
    max_samples: int,
    allow_partial: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    raw_df = read_raw_split(raw_dir, subset, split, max_samples=max_samples)
    labels, source_files = read_label_split(labels_root, model_alias, subset, split)
    records: List[Dict[str, Any]] = []
    missing_uids: List[str] = []

    for raw_row in raw_df.to_dict(orient="records"):
        uid = sample_uid(subset, split, raw_row["id"])
        label = labels.get(uid)
        if label is None:
            missing_uids.append(uid)
            continue
        records.append(build_record(raw_row, label, model_alias, subset, split))

    extra_labels = sorted(set(labels) - {sample_uid(subset, split, row["id"]) for row in raw_df.to_dict(orient="records")})
    if missing_uids and not allow_partial:
        preview = ", ".join(missing_uids[:5])
        raise ValueError(f"Missing {len(missing_uids)} labels for {model_alias}/{subset}/{split}: {preview}")
    if extra_labels and not allow_partial and max_samples <= 0:
        preview = ", ".join(extra_labels[:5])
        raise ValueError(f"Found {len(extra_labels)} labels without matching raw sample for {model_alias}/{subset}/{split}: {preview}")

    stats = {
        "model_alias": model_alias,
        "subset": subset,
        "split": split,
        "raw_records": int(len(raw_df)),
        "label_records": int(len(labels)),
        "output_records": int(len(records)),
        "missing_labels": int(len(missing_uids)),
        "extra_labels": int(len(extra_labels)),
        "label_files": source_files,
        "task_type_counts": pd.Series([record["task_type"] for record in records]).value_counts().sort_index().to_dict(),
        "tool_necessary_counts": pd.Series([record["tool_necessary"] for record in records]).value_counts().sort_index().to_dict(),
    }
    return records, stats


def write_records(records: List[Dict[str, Any]], model_output_dir: Path, subset: str, split: str, overwrite: bool, write_parquet: bool) -> None:
    subset_dir = model_output_dir / subset
    write_jsonl(records, subset_dir / f"{split}.jsonl", overwrite)
    if write_parquet:
        parquet_path = subset_dir / f"{split}.parquet"
        ensure_can_write(parquet_path, overwrite)
        pd.DataFrame(records).to_parquet(parquet_path, index=False)


def summarize_records(records: List[Dict[str, Any]], model_alias: str, subset: str, split: str) -> List[Dict[str, Any]]:
    if not records:
        return []
    df = pd.DataFrame(records)
    keys = ["task_type", "task_type_name", "env_name", "difficulty", "tool_necessary"]
    counts = df.groupby(keys, dropna=False).size().reset_index(name="n")
    rows: List[Dict[str, Any]] = []
    for row in counts.to_dict(orient="records"):
        rows.append({
            "model_alias": model_alias,
            "subset": subset,
            "split": split,
            "task_type": row["task_type"],
            "task_type_name": row["task_type_name"],
            "env_name": row["env_name"],
            "difficulty": row["difficulty"],
            "tool_necessary": int(row["tool_necessary"]),
            "n": int(row["n"]),
        })
    return rows


def write_model_side_files(
    model_output_dir: Path,
    model_alias: str,
    raw_dir_arg: str,
    labels_root_arg: str,
    output_dir_arg: str,
    split_stats: List[Dict[str, Any]],
    summary_rows: List[Dict[str, Any]],
    args: argparse.Namespace,
    write_parquet: bool,
) -> None:
    model_output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = model_output_dir / "summary.csv"
    ensure_can_write(summary_path, args.overwrite)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    coverage_path = model_output_dir / "label_coverage.csv"
    ensure_can_write(coverage_path, args.overwrite)
    pd.DataFrame(split_stats).drop(columns=["label_files"], errors="ignore").to_csv(coverage_path, index=False)

    manifest = {
        "dataset": "modified_when2tool_with_labels",
        "source_dataset": "When2Tool",
        "model_alias": model_alias,
        "raw_dir": raw_dir_arg,
        "labels_root": labels_root_arg,
        "output_dir": output_dir_arg,
        "model_output_dir": str(model_output_dir),
        "allow_partial": bool(args.allow_partial),
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
            *LABEL_COLUMNS,
        ],
        "task_types": TASK_TYPES,
        "env_to_task_type": ENV_TO_TASK_TYPE,
    }
    write_json(model_output_dir / "manifest.json", manifest, args.overwrite)
    write_json(model_output_dir / "env_type_mapping.json", {"task_types": TASK_TYPES, "env_to_task_type": ENV_TO_TASK_TYPE, "rows": mapping_rows()}, args.overwrite)

    readme_path = model_output_dir / "README.md"
    ensure_can_write(readme_path, args.overwrite)
    readme_path.write_text(
        f"# Modified When2Tool Dataset: {model_alias}\n\n"
        "Generated after model-specific When2Tool hard_no_tool labels are available.\n\n"
        "Each row preserves the raw When2Tool fields, adds A/B/C task-type metadata, "
        "and joins the model's `tool_necessary` label plus no-tool answer metadata.\n",
        encoding="utf-8",
    )


def resolve_model_aliases(labels_root: Path, requested: Sequence[str]) -> List[str]:
    if len(requested) == 1 and requested[0] == "all":
        aliases = sorted(path.name for path in labels_root.iterdir() if path.is_dir())
        if not aliases:
            raise FileNotFoundError(f"No model label directories found under {labels_root}")
        return aliases
    return list(requested)


def write_top_level_files(output_dir: Path, model_aliases: Sequence[str], args: argparse.Namespace, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "env_type_mapping.json", {"task_types": TASK_TYPES, "env_to_task_type": ENV_TO_TASK_TYPE, "rows": mapping_rows()}, overwrite)
    write_json(
        output_dir / "manifest.json",
        {
            "dataset": "modified_when2tool_with_labels",
            "model_aliases": list(model_aliases),
            "layout": "<output_dir>/<model_alias>/<subset>/<split>.jsonl",
            "allow_partial": bool(args.allow_partial),
        },
        overwrite,
    )
    baidu_path = output_dir / "baidu_netdisk_info.md"
    if not baidu_path.exists():
        baidu_path.write_text(
            "# Baidu Netdisk Handoff\n\n"
            "resource_name:\nnetdisk_url:\nextract_code:\narchive_name:\n"
            "target_dir: tool_decision_neurons_data/datasets/modified_when2tool/\n"
            "expected_files: manifest.json, env_type_mapping.json, <model_alias>/\n"
            "usage: model-specific modified When2Tool datasets with tool_necessary labels.\n",
            encoding="utf-8",
        )


def build_for_model(
    model_alias: str,
    raw_dir: Path,
    labels_root: Path,
    output_dir: Path,
    raw_dir_arg: str,
    labels_root_arg: str,
    output_dir_arg: str,
    args: argparse.Namespace,
    write_parquet: bool,
) -> List[Dict[str, Any]]:
    model_output_dir = output_dir / model_alias
    split_stats: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for subset in args.subsets:
        for split in args.splits:
            records, stats = build_records_for_split(
                raw_dir=raw_dir,
                labels_root=labels_root,
                model_alias=model_alias,
                subset=subset,
                split=split,
                max_samples=args.max_samples,
                allow_partial=args.allow_partial,
            )
            write_records(records, model_output_dir, subset, split, args.overwrite, write_parquet)
            split_stats.append(stats)
            summary_rows.extend(summarize_records(records, model_alias, subset, split))
            print(
                f"{model_alias}/{subset}/{split}: "
                f"raw={stats['raw_records']} labels={stats['label_records']} "
                f"output={stats['output_records']} missing={stats['missing_labels']} extra={stats['extra_labels']}"
            )

    write_model_side_files(
        model_output_dir=model_output_dir,
        model_alias=model_alias,
        raw_dir_arg=raw_dir_arg,
        labels_root_arg=labels_root_arg,
        output_dir_arg=output_dir_arg,
        split_stats=split_stats,
        summary_rows=summary_rows,
        args=args,
        write_parquet=write_parquet,
    )
    return split_stats


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    raw_dir_arg = args.raw_dir or str(data_root / "datasets" / "raw_when2tool")
    labels_root_arg = args.labels_root or str(data_root / "labels")
    output_dir_arg = args.output_dir or str(data_root / "datasets" / "modified_when2tool")
    raw_dir = Path(raw_dir_arg)
    labels_root = Path(labels_root_arg)
    output_dir = Path(output_dir_arg)
    write_parquet = not args.no_parquet

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset directory not found: {raw_dir}")
    if not labels_root.exists():
        raise FileNotFoundError(f"Labels root not found: {labels_root}")

    model_aliases = resolve_model_aliases(labels_root, args.model_aliases)
    all_stats: List[Dict[str, Any]] = []
    for model_alias in model_aliases:
        all_stats.extend(
            build_for_model(
                model_alias=model_alias,
                raw_dir=raw_dir,
                labels_root=labels_root,
                output_dir=output_dir,
                raw_dir_arg=raw_dir_arg,
                labels_root_arg=labels_root_arg,
                output_dir_arg=output_dir_arg,
                args=args,
                write_parquet=write_parquet,
            )
        )

    write_top_level_files(output_dir, model_aliases, args, args.overwrite)
    print(f"Wrote modified labeled dataset to: {output_dir}")
    print(f"Processed models: {', '.join(model_aliases)}")


if __name__ == "__main__":
    main()
