import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from benchmarking.common import LANGUAGES, read_jsonl, task_id_from_reference, write_json


REQUIRED_COLUMNS = {
    "task_id",
    "path",
    "left_context",
    "right_context",
    "crossfile_context",
    "groundtruth",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate normalized CrossCodeEval data and its repository snapshots"
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--benchmark", default="cceval")
    parser.add_argument(
        "--repos-root", default="data_sources/crosscodeeval_repositories"
    )
    parser.add_argument(
        "--official-root", default="data_sources/crosscodeeval_official/extracted"
    )
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_candidates(value):
    if isinstance(value, str):
        return json.loads(value)
    if value is None:
        return []
    return list(value)


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    repos_root = Path(args.repos_root)
    official_root = Path(args.official_root)
    manifest = {
        "benchmark": args.benchmark,
        "source": "CrossCodeEval official archive (amazon-science/cceval)",
        "languages": {},
    }

    for language in LANGUAGES:
        source_file = official_root / language / "line_completion.jsonl"
        parquet_path = data_root / args.benchmark / language / "test.parquet"
        reference_path = data_root / args.benchmark / language / "test.jsonl"
        excluded_task_ids = []
        local_exclusions_path = data_root / args.benchmark / language / "exclusions.json"
        if local_exclusions_path.exists():
            with local_exclusions_path.open("r", encoding="utf-8") as handle:
                local_excluded_ids = json.load(handle)["task_ids"]
            if local_excluded_ids != excluded_task_ids:
                raise ValueError(f"{language}: local exclusions differ from benchmark configuration")
        excluded_task_id_set = set(excluded_task_ids)
        references = read_jsonl(reference_path)
        if source_file.exists():
            source_rows = read_jsonl(source_file)
            source_description = str(source_file)
        else:
            # The server-ready archive intentionally omits the redundant raw
            # CrossCodeEval JSONL files.  The normalized reference JSONL keeps
            # the same task/repository metadata needed for snapshot checks.
            source_rows = references
            source_description = f"{reference_path} (normalized fallback)"
            print(
                f"{language}: official source is not packaged; "
                f"validating against {reference_path}"
            )
        frame = pd.read_parquet(parquet_path)
        missing_columns = REQUIRED_COLUMNS - set(frame.columns)
        if missing_columns:
            raise ValueError(f"{language}: missing parquet columns {sorted(missing_columns)}")
        source_rows = [
            row for row in source_rows
            if task_id_from_reference(row) not in excluded_task_id_set
        ]
        source_ids = [task_id_from_reference(row) for row in source_rows]
        reference_ids = [task_id_from_reference(row) for row in references]
        parquet_ids = [str(value) for value in frame["task_id"].tolist()]
        if excluded_task_id_set.intersection(reference_ids) or excluded_task_id_set.intersection(parquet_ids):
            raise ValueError(f"{language}: excluded task_id remains in the normalized test set")
        if source_ids != reference_ids or source_ids != parquet_ids:
            raise ValueError(f"{language}: task_id sequence differs from the official source")

        snapshot_kind_by_repository = {}
        for reference in references:
            repository_id = str(reference["metadata"]["repository"])
            if repository_id in snapshot_kind_by_repository:
                continue
            snapshot_path = repos_root / repository_id / "_snapshot.json"
            if not snapshot_path.exists():
                raise FileNotFoundError(f"{language}: missing snapshot metadata: {snapshot_path}")
            with snapshot_path.open("r", encoding="utf-8") as handle:
                snapshot = json.load(handle)
            snapshot_kind_by_repository[repository_id] = snapshot.get(
                "snapshot_kind", "exact_github_revision"
            )

        candidate_counts = []
        for row, reference in zip(frame.itertuples(index=False), references):
            candidates = normalize_candidates(row.crossfile_context)
            candidate_paths = {str(item.get("path", "")).replace("\\", "/") for item in candidates}
            current_path = str(row.path).replace("\\", "/")
            if not candidates:
                raise ValueError(f"{language}: {row.task_id} has no retrieval candidates")
            if current_path in candidate_paths:
                raise ValueError(f"{language}: target file leaks into candidates for {row.task_id}")
            repository_id = str(reference["metadata"]["repository"])
            if snapshot_kind_by_repository[repository_id] == "official_context_fragments_fallback":
                target = str(row.groundtruth)
                if target and any(target in str(item.get("text", "")) for item in candidates):
                    raise ValueError(
                        f"{language}: full target appears in a fallback candidate for {row.task_id}"
                    )
            candidate_counts.append(len(candidates))

        snapshot_kinds = Counter()
        samples_by_snapshot_kind = Counter()
        for row in source_rows:
            repository_id = str(row["metadata"]["repository"])
            kind = snapshot_kind_by_repository[repository_id]
            snapshot_kinds[kind] += 1
            samples_by_snapshot_kind[kind] += 1

        manifest["languages"][language] = {
            "validation_source": source_description,
            "samples": len(frame),
            "repositories": len({str(row["metadata"]["repository"]) for row in source_rows}),
            "average_candidates_per_sample": round(sum(candidate_counts) / len(candidate_counts), 4),
            "minimum_candidates_per_sample": min(candidate_counts),
            "maximum_candidates_per_sample": max(candidate_counts),
            "snapshot_kinds_by_sample": dict(sorted(samples_by_snapshot_kind.items())),
            "parquet_bytes": parquet_path.stat().st_size,
            "jsonl_bytes": reference_path.stat().st_size,
            "parquet_sha256": sha256(parquet_path),
            "jsonl_sha256": sha256(reference_path),
        }
        if excluded_task_ids:
            manifest["languages"][language]["excluded_task_ids"] = excluded_task_ids
        print(
            f"{language}: {len(frame)} samples; "
            f"candidate files/sample={sum(candidate_counts) / len(candidate_counts):.2f}; "
            f"snapshot kinds={dict(snapshot_kinds)}"
        )

    write_json(data_root / args.benchmark / "dataset_manifest.json", manifest)
    print(f"Validation succeeded: {data_root / args.benchmark / 'dataset_manifest.json'}")


if __name__ == "__main__":
    main()
