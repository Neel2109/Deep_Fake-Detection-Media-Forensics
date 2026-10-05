from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Sequence

from PIL import Image

from app.config import ALLOWED_EXTENSIONS
from app.services.xception_detector import LABEL_MAPPING, MIN_CALIBRATION_SAMPLES_PER_CLASS

GROUP_MANIFEST_COLUMNS = {"path", "group_id", "source", "license"}
EVALUATION_SPLITS = {"validation", "calibration", "test"}


def collect_split_examples(
    data_root: Path,
    split_names: Sequence[str],
) -> dict[str, list[tuple[Path, int]]]:
    examples_by_split = {}
    for split_name in split_names:
        split_root = data_root / split_name
        examples: list[tuple[Path, int]] = []
        for label, class_id in LABEL_MAPPING.items():
            class_directory = split_root / label
            if not class_directory.is_dir():
                raise ValueError(f"Dataset class directory does not exist: {class_directory}")
            images = sorted(
                path
                for path in class_directory.rglob("*")
                if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS["image"]
            )
            if not images:
                raise ValueError(f"No supported image files found in {class_directory}")
            examples.extend((path, class_id) for path in images)
        examples_by_split[split_name] = examples
    return examples_by_split


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_group_manifest(
    manifest_path: Path,
    expected_paths: set[str],
) -> tuple[dict[str, dict[str, str]], str]:
    if not manifest_path.is_file():
        raise ValueError(f"Group/provenance manifest does not exist: {manifest_path}")

    rows: dict[str, dict[str, str]] = {}
    try:
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or ())
            missing = GROUP_MANIFEST_COLUMNS.difference(columns)
            if missing:
                raise ValueError(
                    "Group/provenance manifest is missing columns: "
                    + ", ".join(sorted(missing))
                )
            for line_number, row in enumerate(reader, start=2):
                raw_path = (row.get("path") or "").strip().replace("\\", "/")
                relative_path = PurePosixPath(raw_path)
                if (
                    not raw_path
                    or relative_path.is_absolute()
                    or PureWindowsPath(raw_path).is_absolute()
                    or ".." in relative_path.parts
                ):
                    raise ValueError(
                        f"Invalid relative media path in group/provenance manifest line {line_number}"
                    )
                path_key = relative_path.as_posix()
                if path_key in rows:
                    raise ValueError(
                        f"Duplicate path in group/provenance manifest line {line_number}: {path_key}"
                    )
                record = {
                    column: (row.get(column) or "").strip()
                    for column in GROUP_MANIFEST_COLUMNS
                    if column != "path"
                }
                if any(not value for value in record.values()):
                    raise ValueError(
                        f"Manifest line {line_number} requires non-empty group_id, source, and license"
                    )
                rows[path_key] = record
    except OSError as error:
        raise ValueError(f"Could not read group/provenance manifest: {error}") from error

    missing_paths = expected_paths.difference(rows)
    unknown_paths = set(rows).difference(expected_paths)
    if missing_paths or unknown_paths:
        raise ValueError(
            "Group/provenance manifest must cover every dataset image exactly once "
            f"(missing={len(missing_paths)}, unknown={len(unknown_paths)})"
        )
    return rows, _sha256_file(manifest_path)


def audit_dataset(
    data_root: Path,
    split_examples: Mapping[str, Sequence[tuple[Path, int]]],
    group_manifest_path: Path | None = None,
) -> dict[str, Any]:
    root = data_root.resolve(strict=True)
    labels_by_id = {class_id: label for label, class_id in LABEL_MAPPING.items()}
    files: list[dict[str, Any]] = []
    split_counts: dict[str, dict[str, int]] = {}
    paths_seen: set[str] = set()
    hashes_by_split: dict[str, dict[str, list[str]]] = {}

    for split_name, examples in split_examples.items():
        counts = {label: 0 for label in LABEL_MAPPING}
        hashes_by_split[split_name] = {}
        for path, label_id in examples:
            resolved = path.resolve(strict=True)
            try:
                relative_path = resolved.relative_to(root).as_posix()
            except ValueError as error:
                raise ValueError(f"Dataset image resolves outside its root: {path}") from error
            if relative_path in paths_seen:
                raise ValueError(f"Dataset image is listed more than once: {relative_path}")
            paths_seen.add(relative_path)
            if label_id not in labels_by_id:
                raise ValueError(f"Unsupported dataset label {label_id!r} for {relative_path}")

            try:
                with Image.open(resolved) as image:
                    if getattr(image, "is_animated", False):
                        raise ValueError(f"Animated image is not supported for training: {relative_path}")
                    width, height = image.size
                    image_format = image.format or "unknown"
                    image.load()
            except (OSError, SyntaxError, ValueError) as error:
                raise ValueError(f"Could not decode dataset image {relative_path}: {error}") from error

            digest = _sha256_file(resolved)
            hashes_by_split[split_name].setdefault(digest, []).append(relative_path)
            counts[labels_by_id[label_id]] += 1
            files.append({
                "path": relative_path,
                "split": split_name,
                "label": labels_by_id[label_id],
                "sha256": digest,
                "size_bytes": resolved.stat().st_size,
                "width": width,
                "height": height,
                "format": image_format,
            })
        split_counts[split_name] = counts

    duplicate_digests: dict[str, dict[str, list[str]]] = {}
    training_duplicate_groups = []
    for split_name, digest_paths in hashes_by_split.items():
        for digest, paths in digest_paths.items():
            duplicate_digests.setdefault(digest, {})[split_name] = paths
    for digest, split_paths in duplicate_digests.items():
        split_names = sorted(split_paths)
        if len(split_names) == 1 and len(split_paths[split_names[0]]) == 1:
            continue
        if len(split_names) > 1:
            raise ValueError(
                f"Found byte-identical image(s) shared by {' and '.join(split_names)} splits "
                f"(SHA-256 {digest[:12]})"
            )
        split_name = split_names[0]
        if split_name in EVALUATION_SPLITS:
            raise ValueError(
                f"Found byte-identical duplicate images inside the {split_name} evaluation split "
                f"(SHA-256 {digest[:12]})"
            )
        training_duplicate_groups.append({
            "sha256": digest,
            "count": len(split_paths[split_name]),
            "paths": split_paths[split_name],
        })

    expected_paths = {item["path"] for item in files}
    group_manifest_hash = None
    group_overlap_status = "not_checked"
    source_splits: dict[str, set[str]] = {}
    warnings = [
        "Source/person group leakage could not be checked because no group/provenance manifest was supplied.",
        "Exact-hash checks do not detect visually similar or transformed copies.",
    ]
    if group_manifest_path is not None:
        group_manifest, group_manifest_hash = _load_group_manifest(
            group_manifest_path,
            expected_paths,
        )
        group_to_splits: dict[str, set[str]] = {}
        for item in files:
            record = group_manifest[item["path"]]
            item.update(record)
            group_to_splits.setdefault(record["group_id"], set()).add(item["split"])
            source_splits.setdefault(record["source"], set()).add(item["split"])
        overlaps = {
            group_id: sorted(splits)
            for group_id, splits in group_to_splits.items()
            if len(splits) > 1
        }
        if overlaps:
            example = next(iter(overlaps.items()))
            raise ValueError(
                f"Source/person group leakage found across splits: group {example[0]!r} "
                f"appears in {', '.join(example[1])}; repartition by group before training"
            )
        group_overlap_status = "checked_no_overlap"
        warnings = [
            "Group overlap checks rely on the accuracy and completeness of the supplied manifest.",
            "Exact-hash checks do not detect visually similar or transformed copies.",
        ]
    if training_duplicate_groups:
        warnings.append(
            f"{len(training_duplicate_groups)} exact-duplicate group(s) in the training split may overweight examples."
        )

    report: dict[str, Any] = {
        "format_version": 1,
        "label_mapping": dict(LABEL_MAPPING),
        "split_counts": split_counts,
        "files": sorted(files, key=lambda item: (item["split"], item["path"])),
        "training_duplicate_groups": training_duplicate_groups,
        "group_manifest_sha256": group_manifest_hash,
        "group_overlap_status": group_overlap_status,
        "source_split_summary": {
            source: sorted(splits)
            for source, splits in sorted(source_splits.items())
        },
        "warnings": warnings,
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    report["manifest_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return report


def write_audit_report(report: Mapping[str, Any], output_path: Path) -> Path:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(output_path)
    except OSError:
        temporary_path.unlink(missing_ok=True)
        raise
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit an Xception dataset for decode failures and split leakage."
    )
    parser.add_argument("--data-root", required=True, help="Directory containing train/, validation/, calibration/, and optional test/")
    parser.add_argument(
        "--group-manifest",
        help="CSV with path, group_id, source, and license columns; paths are relative to data-root",
    )
    parser.add_argument(
        "--output",
        help="Audit JSON output path (defaults to <data-root>/dataset-audit.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data_root = Path(args.data_root).resolve()
    split_names = ["train", "validation", "calibration"]
    if (data_root / "test").is_dir():
        split_names.append("test")
    split_examples = collect_split_examples(data_root, split_names)
    split_counts = {
        split_name: {
            label: sum(class_id == label_id for _, class_id in examples)
            for label, label_id in LABEL_MAPPING.items()
        }
        for split_name, examples in split_examples.items()
    }
    if any(
        count == 0
        for split_name in ("train", "validation")
        for count in split_counts[split_name].values()
    ):
        raise ValueError("Both real and fake examples are required in train and validation splits")
    if any(
        count < MIN_CALIBRATION_SAMPLES_PER_CLASS
        for count in split_counts["calibration"].values()
    ):
        raise ValueError(
            "Calibration split requires at least "
            f"{MIN_CALIBRATION_SAMPLES_PER_CLASS} images in each class"
        )
    if "test" in split_counts and any(
        count < MIN_CALIBRATION_SAMPLES_PER_CLASS
        for count in split_counts["test"].values()
    ):
        raise ValueError(
            "Independent test split requires at least "
            f"{MIN_CALIBRATION_SAMPLES_PER_CLASS} images in each class"
        )

    group_manifest_path = None
    if args.group_manifest:
        candidate = Path(args.group_manifest)
        group_manifest_path = candidate if candidate.is_absolute() else data_root / candidate
    report = audit_dataset(data_root, split_examples, group_manifest_path)
    output_path = (
        Path(args.output)
        if args.output
        else data_root / "dataset-audit.json"
    )
    saved_path = write_audit_report(report, output_path)
    print(json.dumps({
        "audit_report": str(saved_path),
        "manifest_sha256": report["manifest_sha256"],
        "group_overlap_status": report["group_overlap_status"],
        "split_counts": report["split_counts"],
        "warnings": report["warnings"],
    }, indent=2))


if __name__ == "__main__":
    main()
