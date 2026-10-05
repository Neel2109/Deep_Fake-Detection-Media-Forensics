import importlib.util
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from PIL import Image

from app.config import XCEPTION_CHECKPOINT
from app.services.xception_detector import (
    ARCHITECTURE,
    IMAGE_SIZE,
    LABEL_MAPPING,
    MIN_CALIBRATION_SAMPLES_PER_CLASS,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
    PUBLIC_ARCHITECTURE,
    XceptionDetector,
    XceptionModelError,
    _gradcam_explanation,
    detect_largest_face_crop,
    get_xception_detector,
    validate_checkpoint_contract,
)
from training.train_xception import calculate_auc, select_operating_thresholds


def valid_checkpoint():
    return {
        "architecture": ARCHITECTURE,
        "image_size": IMAGE_SIZE,
        "num_logits": 1,
        "label_mapping": LABEL_MAPPING,
        "normalization": {
            "mean": list(NORMALIZATION_MEAN),
            "std": list(NORMALIZATION_STD),
        },
        "model_state_dict": {"weight": "test-only placeholder"},
        "best_epoch": 1,
        "validation_auc": 0.91,
        "calibration": {
            "status": "temperature_scaled",
            "temperature": 1.4,
            "method": "single scalar optimized with held-out BCE",
            "samples_per_class": {
                "real": MIN_CALIBRATION_SAMPLES_PER_CLASS,
                "fake": MIN_CALIBRATION_SAMPLES_PER_CLASS,
            },
        },
        "thresholds": {
            "real_max_probability": 0.2,
            "fake_min_probability": 0.8,
            "calibration_operating_point": 0.90,
            "status": "available",
        },
    }


class XceptionContractTests(unittest.TestCase):
    def test_checkpoint_contract_accepts_expected_metadata(self):
        validate_checkpoint_contract(valid_checkpoint())

    def test_checkpoint_contract_rejects_wrong_label_mapping(self):
        checkpoint = valid_checkpoint()
        checkpoint["label_mapping"] = {"real": 1, "fake": 0}
        with self.assertRaisesRegex(XceptionModelError, "real=0 and fake=1"):
            validate_checkpoint_contract(checkpoint)

    def test_checkpoint_contract_rejects_missing_calibration_samples(self):
        checkpoint = valid_checkpoint()
        checkpoint["calibration"]["samples_per_class"]["fake"] = (
            MIN_CALIBRATION_SAMPLES_PER_CLASS - 1
        )
        with self.assertRaisesRegex(XceptionModelError, "at least"):
            validate_checkpoint_contract(checkpoint)

    def test_checkpoint_contract_rejects_overlapping_operating_thresholds(self):
        checkpoint = valid_checkpoint()
        checkpoint["thresholds"] = {
            "real_max_probability": 0.8,
            "fake_min_probability": 0.7,
            "calibration_operating_point": 0.90,
            "status": "available",
        }
        with self.assertRaisesRegex(XceptionModelError, "non-overlapping"):
            validate_checkpoint_contract(checkpoint)

    def test_checkpoint_contract_allows_calibrated_probability_without_safe_gap(self):
        checkpoint = valid_checkpoint()
        checkpoint["thresholds"] = {
            "real_max_probability": None,
            "fake_min_probability": None,
            "calibration_operating_point": 0.90,
            "status": "no_non_overlapping_operating_gap",
        }
        validate_checkpoint_contract(checkpoint)

    def test_checkpoint_contract_rejects_invalid_optional_conformal_metadata(self):
        checkpoint = valid_checkpoint()
        checkpoint["calibration"]["conformal_prediction"] = {
            "method": "class_conditional_split_conformal",
            "alpha": 0.1,
            "quantile_real_nonconformity": 1.2,
            "quantile_fake_nonconformity": 0.3,
            "samples_per_class": {"real": 20, "fake": 20},
            "calibration_scope": "test scope",
        }

        with self.assertRaisesRegex(XceptionModelError, "conformal calibration metadata"):
            validate_checkpoint_contract(checkpoint)

    def test_checkpoint_contract_rejects_validation_auc_at_chance(self):
        checkpoint = valid_checkpoint()
        checkpoint["validation_auc"] = 0.5
        with self.assertRaisesRegex(XceptionModelError, "greater than chance"):
            validate_checkpoint_contract(checkpoint)

    def test_operating_thresholds_use_held_out_scores_and_leave_a_gap(self):
        result = select_operating_thresholds(
            [0.05, 0.15, 0.80, 0.90],
            [0, 0, 1, 1],
            minimum_sensitivity_specificity=0.90,
        )
        self.assertEqual(result["status"], "available")
        self.assertLess(result["real_max_probability"], result["fake_min_probability"])
        self.assertEqual(result["real_threshold_observed_sensitivity"], 1.0)
        self.assertEqual(result["fake_threshold_observed_specificity"], 1.0)

    def test_no_safe_threshold_gap_keeps_classifier_inconclusive(self):
        result = select_operating_thresholds(
            [0.4, 0.6, 0.4, 0.6],
            [0, 0, 1, 1],
            minimum_sensitivity_specificity=0.90,
        )
        self.assertEqual(result["status"], "no_non_overlapping_operating_gap")
        self.assertIsNone(result["real_max_probability"])
        self.assertIsNone(result["fake_min_probability"])

    def test_auc_handles_ties_and_requires_both_classes(self):
        self.assertEqual(calculate_auc([0.2, 0.2, 0.8, 0.8], [0, 1, 0, 1]), 0.5)
        with self.assertRaisesRegex(ValueError, "each class"):
            calculate_auc([0.2, 0.8], [0, 0])

    def test_face_detector_uses_full_image_fallback_when_detection_is_unavailable(self):
        image = Image.new("RGB", (64, 64), "white")
        crop, review = detect_largest_face_crop(image)
        self.assertEqual(crop.size, image.size)
        self.assertIn(review["status"], {"no_face_detected", "unavailable"})
        self.assertEqual(review["selected_crop"], "full_image_fallback")
        self.assertEqual(review["alignment"], "not_performed_landmarks_unavailable")

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch is not installed")
    def test_gradcam_generates_bounded_overlay_for_fake_class_logit(self):
        torch = importlib.import_module("torch")
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(4, 2),
        )
        model.eval()
        explanation = _gradcam_explanation(
            model,
            torch.rand((1, 3, 24, 32)),
            Image.new("RGB", (800, 400), "navy"),
            torch,
        )

        self.assertEqual(explanation["status"], "generated")
        self.assertEqual(explanation["method"], "Grad-CAM")
        self.assertEqual(explanation["target"], "fake-class logit")
        self.assertLessEqual(max(explanation["overlay_width"], explanation["overlay_height"]), 512)
        self.assertTrue(explanation["overlay_data_url"].startswith("data:image/jpeg;base64,"))

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch is not installed")
    def test_gradcam_reports_not_generated_when_model_has_no_convolution_layer(self):
        torch = importlib.import_module("torch")
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 4 * 4, 2))

        explanation = _gradcam_explanation(
            model,
            torch.rand((1, 3, 4, 4)),
            Image.new("RGB", (4, 4), "navy"),
            torch,
        )

        self.assertEqual(explanation["status"], "not_generated")
        self.assertIn("no convolution layer", explanation["reason"])

    def test_prediction_exposes_dataset_level_checkpoint_metrics(self):
        class FakeTensor:
            def reshape(self, *_shape):
                return self

            def __getitem__(self, _index):
                return self

            def __truediv__(self, _value):
                return self

            def item(self):
                return 0.1

            def unsqueeze(self, _dimension):
                return self

            def to(self, _device):
                return self

        class FakeModel:
            def parameters(self):
                return iter([SimpleNamespace(device="cpu")])

            def __call__(self, _tensor):
                return FakeTensor()

        class FakeTransforms:
            def Compose(self, _transforms):
                return lambda _image: FakeTensor()

            def Resize(self, _size):
                return object()

            def ToTensor(self):
                return object()

            def Normalize(self, _mean, _std):
                return object()

        checkpoint = valid_checkpoint()
        expected_test_metrics = {"samples": 40, "roc_auc": 0.93}
        expected_source_metrics = {"heldout": {"samples": 40, "roc_auc": 0.93}}
        expected_audit = {
            "manifest_sha256": "audit-digest",
            "group_overlap_status": "checked_no_overlap",
        }
        checkpoint["validation_metrics"] = {"samples": 60, "roc_auc": 0.91}
        checkpoint["validation_metrics_temperature_scaled"] = {"samples": 60, "roc_auc": 0.91}
        checkpoint["test_metrics"] = expected_test_metrics
        checkpoint["test_metrics_by_source"] = expected_source_metrics
        checkpoint["dataset_audit"] = expected_audit
        checkpoint["calibration"]["metrics_before_scaling"] = {"brier_score": 0.2}
        checkpoint["calibration"]["metrics_after_scaling"] = {"brier_score": 0.1}
        checkpoint["calibration"]["conformal_prediction"] = {
            "method": "class_conditional_split_conformal",
            "alpha": 0.1,
            "quantile_real_nonconformity": 0.01,
            "quantile_fake_nonconformity": 0.01,
            "samples_per_class": {"real": 20, "fake": 20},
            "calibration_scope": "test exchangeability scope",
        }

        detector = XceptionDetector(
            FakeModel(),
            checkpoint,
            SimpleNamespace(inference_mode=nullcontext, sigmoid=lambda tensor: tensor),
            FakeTransforms(),
        )
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "input.png"
            Image.new("RGB", (64, 64), "white").save(image_path)
            result = detector.predict(image_path)

        self.assertEqual(result["model"]["validation_metrics"], checkpoint["validation_metrics"])
        self.assertEqual(result["model"]["validation_metrics_temperature_scaled"], checkpoint["validation_metrics_temperature_scaled"])
        self.assertEqual(result["model"]["test_metrics"], expected_test_metrics)
        self.assertEqual(result["model"]["test_metrics_by_source"], expected_source_metrics)
        self.assertEqual(result["model"]["dataset_audit"], expected_audit)
        self.assertEqual(
            result["model"]["calibration_metrics_before_scaling"],
            checkpoint["calibration"]["metrics_before_scaling"],
        )
        self.assertEqual(
            result["model"]["calibration_metrics_after_scaling"],
            checkpoint["calibration"]["metrics_after_scaling"],
        )
        self.assertEqual(result["status"], "conformal_abstention")
        self.assertEqual(result["verdict"], "Insufficient evidence")
        self.assertEqual(result["conformal_prediction"]["prediction_set"], [])
        self.assertTrue(result["conformal_prediction"]["abstained"])


class PublicXceptionCheckpointSmokeTests(unittest.TestCase):
    @unittest.skipUnless(
        XCEPTION_CHECKPOINT.is_file()
        and importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("torchvision") is not None,
        "The optional public checkpoint and PyTorch runtime are not installed",
    )
    def test_pinned_checkpoint_loads_and_produces_normalized_uncalibrated_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "smoke.png"
            Image.new("RGB", (320, 240), (132, 88, 170)).save(image_path)
            detector = get_xception_detector(XCEPTION_CHECKPOINT)
            self.assertIsNotNone(detector)
            result = detector.predict(image_path)

        self.assertEqual(result["model"]["architecture"], PUBLIC_ARCHITECTURE)
        self.assertEqual(result["status"], "uncalibrated_prediction")
        self.assertAlmostEqual(
            result["deepfake_probability"] + result["authenticity_probability"],
            1.0,
            places=5,
        )
        self.assertEqual(result["model"]["class_mapping"], {"fake": 0, "real": 1})
        self.assertEqual(result["face_review"]["model_input"], "full_image")
        self.assertEqual(result["explainability"]["status"], "generated")


if __name__ == "__main__":
    unittest.main()
