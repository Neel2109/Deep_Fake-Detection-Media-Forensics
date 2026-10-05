import csv
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from training.dataset_audit import audit_dataset, write_audit_report
from training.train_xception import (
    _classification_metrics,
    _operating_point_metrics,
    calculate_average_precision,
    calculate_class_conditional_conformal_thresholds,
)


class BenchmarkMetricTests(unittest.TestCase):
    def test_average_precision_is_one_for_perfect_ranked_predictions(self):
        self.assertEqual(
            calculate_average_precision([0.1, 0.9, 0.2, 0.8], [0, 1, 0, 1]),
            1.0,
        )

    def test_average_precision_groups_tied_scores_at_the_same_threshold(self):
        self.assertEqual(
            calculate_average_precision([0.8, 0.8, 0.2, 0.2], [0, 1, 0, 1]),
            0.5,
        )

    def test_classification_metrics_include_balanced_and_calibration_metrics(self):
        result = _classification_metrics([0.1, 0.7, 0.8, 0.2], [0, 0, 1, 1])

        self.assertEqual(result["confusion_matrix_real_fake"], [[1, 1], [1, 1]])
        self.assertEqual(result["balanced_accuracy_at_0_5"], 0.5)
        self.assertEqual(result["f1_at_0_5"], 0.5)
        self.assertAlmostEqual(result["brier_score"], 0.295)
        self.assertAlmostEqual(result["expected_calibration_error"], 0.45)
        self.assertEqual(result["samples_per_class"], {"real": 2, "fake": 2})

    def test_selective_metrics_report_abstention_coverage(self):
        result = _operating_point_metrics(
            [0.1, 0.5, 0.9],
            [0, 1, 1],
            real_max_probability=0.2,
            fake_min_probability=0.8,
        )

        self.assertEqual(result["coverage"], 2 / 3)
        self.assertEqual(result["abstained_samples"], 1)
        self.assertEqual(result["accuracy_on_decided_samples"], 1.0)

    def test_class_conditional_conformal_thresholds_use_finite_sample_quantiles(self):
        result = calculate_class_conditional_conformal_thresholds(
            [0.1, 0.2, 0.8, 0.9],
            [0, 0, 1, 1],
            alpha=0.25,
        )

        self.assertEqual(result["method"], "class_conditional_split_conformal")
        self.assertEqual(result["quantile_real_nonconformity"], 0.2)
        self.assertAlmostEqual(result["quantile_fake_nonconformity"], 0.2)
        self.assertEqual(result["samples_per_class"], {"real": 2, "fake": 2})

    def test_conformal_calibration_rejects_missing_class_and_invalid_scores(self):
        with self.assertRaisesRegex(ValueError, "each class"):
            calculate_class_conditional_conformal_thresholds([0.2, 0.3], [0, 0])
        with self.assertRaisesRegex(ValueError, "finite probabilities"):
            calculate_class_conditional_conformal_thresholds([float("nan"), 0.3], [0, 1])
        with self.assertRaisesRegex(ValueError, "alpha"):
            calculate_class_conditional_conformal_thresholds([0.2, 0.8], [0, 1], alpha=1.0)


class DatasetAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)

    def _create_example(self, relative_path: str, color: tuple[int, int, int]) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (24, 24), color).save(path)
        return path

    def _example_splits(self):
        return {
            "train": [
                (self._create_example("train/real/real.png", (10, 10, 10)), 0),
                (self._create_example("train/fake/fake.png", (240, 10, 10)), 1),
            ],
            "validation": [
                (self._create_example("validation/real/real.png", (10, 10, 30)), 0),
                (self._create_example("validation/fake/fake.png", (240, 10, 30)), 1),
            ],
        }

    def _write_group_manifest(self, splits, groups=None) -> Path:
        path = self.root / "groups.csv"
        groups = groups or {}
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["path", "group_id", "source", "license"],
            )
            writer.writeheader()
            for split, examples in splits.items():
                for example_path, _ in examples:
                    relative = example_path.relative_to(self.root).as_posix()
                    writer.writerow({
                        "path": relative,
                        "group_id": groups.get(relative, relative),
                        "source": "approved-test-corpus",
                        "license": "CC0-1.0",
                    })
        return path

    def test_audit_records_hashes_and_discloses_missing_group_manifest(self):
        splits = self._example_splits()
        report = audit_dataset(self.root, splits)

        self.assertEqual(report["group_overlap_status"], "not_checked")
        self.assertEqual(report["split_counts"]["train"], {"real": 1, "fake": 1})
        self.assertEqual(len(report["manifest_sha256"]), 64)
        self.assertTrue(any("group leakage" in warning for warning in report["warnings"]))
        self.assertTrue(all(len(item["sha256"]) == 64 for item in report["files"]))

    def test_manifest_checks_source_groups_across_splits(self):
        splits = self._example_splits()
        groups = {
            "train/real/real.png": "person-1",
            "validation/fake/fake.png": "person-1",
        }
        manifest = self._write_group_manifest(splits, groups)

        with self.assertRaisesRegex(ValueError, "group leakage"):
            audit_dataset(self.root, splits, manifest)

    def test_manifest_can_confirm_no_source_group_overlap(self):
        splits = self._example_splits()
        manifest = self._write_group_manifest(splits)

        report = audit_dataset(self.root, splits, manifest)

        self.assertEqual(report["group_overlap_status"], "checked_no_overlap")
        self.assertEqual(
            report["source_split_summary"],
            {"approved-test-corpus": ["train", "validation"]},
        )
        self.assertTrue(all("license" in item and "source" in item for item in report["files"]))

    def test_exact_duplicate_across_splits_is_rejected(self):
        original = self._create_example("train/real/original.png", (35, 40, 45))
        copy = self.root / "validation/real/copy.png"
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(original.read_bytes())
        splits = {
            "train": [(original, 0)],
            "validation": [(copy, 0)],
        }

        with self.assertRaisesRegex(ValueError, "byte-identical image"):
            audit_dataset(self.root, splits)

    def test_corrupt_image_is_rejected_during_audit(self):
        corrupt = self.root / "train/real/broken.png"
        corrupt.parent.mkdir(parents=True, exist_ok=True)
        corrupt.write_bytes(b"not an image")

        with self.assertRaisesRegex(ValueError, "Could not decode dataset image"):
            audit_dataset(self.root, {"train": [(corrupt, 0)]})

    def test_audit_report_is_written_with_its_manifest_digest(self):
        splits = self._example_splits()
        report = audit_dataset(self.root, splits)
        output = write_audit_report(report, self.root / "reports" / "audit.json")

        stored_report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(stored_report["manifest_sha256"], report["manifest_sha256"])
        self.assertFalse(output.with_name(f".{output.name}.tmp").exists())


if __name__ == "__main__":
    unittest.main()
