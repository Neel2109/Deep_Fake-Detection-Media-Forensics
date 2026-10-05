from __future__ import annotations

import base64
from collections.abc import Mapping
from functools import lru_cache
from importlib import import_module
from io import BytesIO
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

ARCHITECTURE = "timm:legacy_xception"
PUBLIC_ARCHITECTURE = "RamadhanZome/deepfake-xception"
PUBLIC_MODEL_REVISION = "821a763a28cc8addf665a5be92f5c220f94f0ab3"
PUBLIC_CHECKPOINT_SHA256 = "cbe132b4e674a35eda661731eb6c2a5fb00803b4ba55eb5641664439b9a07139"
IMAGE_SIZE = 299
NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
NORMALIZATION_STD = (0.229, 0.224, 0.225)
LABEL_MAPPING = {"real": 0, "fake": 1}
MIN_CALIBRATION_SAMPLES_PER_CLASS = 20


class XceptionModelError(RuntimeError):
    """A configured Xception detector could not be used safely."""


def validate_checkpoint_contract(checkpoint: Mapping[str, Any]) -> None:
    required = {
        "architecture",
        "image_size",
        "num_logits",
        "label_mapping",
        "normalization",
        "model_state_dict",
        "best_epoch",
        "validation_auc",
        "calibration",
        "thresholds",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise XceptionModelError(
            f"Checkpoint is missing required metadata: {', '.join(sorted(missing))}"
        )
    if checkpoint["architecture"] != ARCHITECTURE:
        raise XceptionModelError(
            f"Unsupported architecture {checkpoint['architecture']!r}; expected {ARCHITECTURE!r}"
        )
    if checkpoint["image_size"] != IMAGE_SIZE or checkpoint["num_logits"] != 1:
        raise XceptionModelError("Checkpoint input size or binary-logit contract is incompatible")
    if checkpoint["label_mapping"] != LABEL_MAPPING:
        raise XceptionModelError("Checkpoint must map real=0 and fake=1")

    normalization = checkpoint["normalization"]
    if not isinstance(normalization, Mapping):
        raise XceptionModelError("Checkpoint normalization metadata must be an object")
    mean = normalization.get("mean")
    std = normalization.get("std")
    if not _matches_values(mean, NORMALIZATION_MEAN) or not _matches_values(std, NORMALIZATION_STD):
        raise XceptionModelError("Checkpoint normalization does not match the ImageNet input contract")

    if not isinstance(checkpoint["model_state_dict"], Mapping) or not checkpoint["model_state_dict"]:
        raise XceptionModelError("Checkpoint model_state_dict is missing or empty")
    if (
        not isinstance(checkpoint["best_epoch"], int)
        or isinstance(checkpoint["best_epoch"], bool)
        or checkpoint["best_epoch"] < 1
    ):
        raise XceptionModelError("Checkpoint best_epoch must be a positive integer")
    validation_auc = checkpoint["validation_auc"]
    if not _is_probability(validation_auc) or validation_auc <= 0.5:
        raise XceptionModelError("Checkpoint validation_auc must be finite and greater than chance (0.5)")

    calibration = checkpoint["calibration"]
    if not isinstance(calibration, Mapping) or calibration.get("status") != "temperature_scaled":
        raise XceptionModelError("Checkpoint has no completed held-out temperature calibration")
    temperature = calibration.get("temperature")
    if not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
        raise XceptionModelError("Checkpoint calibration temperature must be finite and greater than zero")
    if calibration.get("method") != "single scalar optimized with held-out BCE":
        raise XceptionModelError("Checkpoint calibration method is not the supported held-out temperature scaler")
    samples_per_class = calibration.get("samples_per_class")
    if (
        not isinstance(samples_per_class, Mapping)
        or any(
            not isinstance(samples_per_class.get(label), int)
            or samples_per_class[label] < MIN_CALIBRATION_SAMPLES_PER_CLASS
            for label in LABEL_MAPPING
        )
    ):
        raise XceptionModelError(
            f"Checkpoint calibration requires at least {MIN_CALIBRATION_SAMPLES_PER_CLASS} "
            "held-out examples from each class"
        )
    conformal = calibration.get("conformal_prediction")
    if conformal is not None:
        conformal_samples = conformal.get("samples_per_class") if isinstance(conformal, Mapping) else None
        if (
            not isinstance(conformal, Mapping)
            or conformal.get("method") != "class_conditional_split_conformal"
            or not isinstance(conformal.get("alpha"), (int, float))
            or isinstance(conformal.get("alpha"), bool)
            or not math.isfinite(conformal["alpha"])
            or not 0 < conformal["alpha"] < 1
            or not _is_probability(conformal.get("quantile_real_nonconformity"))
            or not _is_probability(conformal.get("quantile_fake_nonconformity"))
            or not isinstance(conformal.get("calibration_scope"), str)
            or not isinstance(conformal_samples, Mapping)
            or any(
                not isinstance(conformal_samples.get(label), int)
                or isinstance(conformal_samples.get(label), bool)
                or conformal_samples[label] < MIN_CALIBRATION_SAMPLES_PER_CLASS
                for label in LABEL_MAPPING
            )
        ):
            raise XceptionModelError("Checkpoint conformal calibration metadata is invalid")

    thresholds = checkpoint["thresholds"]
    if not isinstance(thresholds, Mapping):
        raise XceptionModelError("Checkpoint thresholds metadata must be an object")
    threshold_fields = {
        "real_max_probability",
        "fake_min_probability",
        "calibration_operating_point",
        "status",
    }
    missing_threshold_fields = threshold_fields.difference(thresholds)
    if missing_threshold_fields:
        raise XceptionModelError(
            f"Checkpoint thresholds are missing: {', '.join(sorted(missing_threshold_fields))}"
        )
    operating_point = thresholds.get("calibration_operating_point")
    if not _is_probability(operating_point) or operating_point == 0:
        raise XceptionModelError("Checkpoint must record a positive calibration operating point")
    if thresholds.get("status") not in {"available", "no_non_overlapping_operating_gap"}:
        raise XceptionModelError("Checkpoint operating-threshold status is missing or unsupported")
    real_max = thresholds.get("real_max_probability")
    fake_min = thresholds.get("fake_min_probability")
    if (real_max is not None) != (thresholds["status"] == "available"):
        raise XceptionModelError("Checkpoint threshold values and availability status are inconsistent")
    if (real_max is None) != (fake_min is None):
        raise XceptionModelError("Both operating thresholds must be present, or both must be null")
    if real_max is not None and (
        not _is_probability(real_max)
        or not _is_probability(fake_min)
        or real_max >= fake_min
    ):
        raise XceptionModelError("Operating thresholds must be probabilities with a non-overlapping gap")


def _matches_values(values: Any, expected: tuple[float, ...]) -> bool:
    return (
        isinstance(values, (list, tuple))
        and len(values) == len(expected)
        and all(
            isinstance(actual, (int, float))
            and math.isfinite(actual)
            and math.isclose(actual, target, rel_tol=0, abs_tol=1e-7)
            for actual, target in zip(values, expected)
        )
    )


def _is_probability(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0 <= value <= 1
    )


def _load_face_cascade() -> cv2.CascadeClassifier | None:
    if (
        not hasattr(cv2, "CascadeClassifier")
        or not hasattr(cv2, "data")
        or not hasattr(cv2.data, "haarcascades")
    ):
        return None
    try:
        cascade = cv2.CascadeClassifier(
            str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
        )
    except (AttributeError, OSError, cv2.error):
        return None
    return cascade if not cascade.empty() else None


def detect_largest_face_crop(
    image: Image.Image,
    face_cascade: cv2.CascadeClassifier | None = None,
) -> tuple[Image.Image, dict[str, Any]]:
    rgb_image = image.convert("RGB")
    cascade = face_cascade if face_cascade is not None else _load_face_cascade()
    if cascade is None or cascade.empty():
        return rgb_image, {
            "detector": "OpenCV Haar cascade",
            "status": "unavailable",
            "detected_face_count": 0,
            "face_candidate_boxes_xyxy": [],
            "selected_crop": "full_image_fallback",
            "alignment": "not_performed_landmarks_unavailable",
        }

    gray = cv2.cvtColor(np.asarray(rgb_image), cv2.COLOR_RGB2GRAY)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(48, 48),
    )
    if len(faces) == 0:
        return rgb_image, {
            "detector": "OpenCV Haar cascade",
            "status": "no_face_detected",
            "detected_face_count": 0,
            "face_candidate_boxes_xyxy": [],
            "selected_crop": "full_image_fallback",
            "alignment": "not_performed_landmarks_unavailable",
        }

    x, y, width, height = max(faces, key=lambda face: int(face[2]) * int(face[3]))
    padding_x = int(width * 0.2)
    padding_y = int(height * 0.2)
    left = max(0, int(x) - padding_x)
    top = max(0, int(y) - padding_y)
    right = min(rgb_image.width, int(x + width) + padding_x)
    bottom = min(rgb_image.height, int(y + height) + padding_y)
    return rgb_image.crop((left, top, right, bottom)), {
        "detector": "OpenCV Haar cascade",
        "status": "face_detected",
        "detected_face_count": len(faces),
        "face_candidate_boxes_xyxy": [
            [int(x), int(y), int(x + width), int(y + height)]
            for x, y, width, height in faces
        ],
        "selected_crop": "largest_detected_face_with_20_percent_context",
        "crop_box_xyxy": [left, top, right, bottom],
        "alignment": "not_performed_landmarks_unavailable",
    }


def _gradcam_explanation(
    model: Any,
    model_input: Any,
    source_image: Image.Image,
    torch: Any,
    target_class_index: int = 0,
) -> dict[str, Any]:
    """Build a Grad-CAM overlay for the selected model logit and model input."""
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        return {
            "status": "not_generated",
            "method": "Grad-CAM",
            "reason": "The loaded model does not expose convolution layers for Grad-CAM.",
        }
    convolution_layers = [
        (name, module)
        for name, module in named_modules()
        if isinstance(module, torch.nn.Conv2d)
    ]
    if not convolution_layers:
        return {
            "status": "not_generated",
            "method": "Grad-CAM",
            "reason": "The loaded model exposes no convolution layer for Grad-CAM.",
        }

    width, height = source_image.size
    scale = min(1.0, 512 / max(width, height))
    output_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    layer_name, target_layer = convolution_layers[-1]
    activation: list[Any] = []

    def capture_activation(_module: Any, _inputs: Any, output: Any) -> None:
        if isinstance(output, tuple):
            output = output[0]
        activation.append(output)

    hook = target_layer.register_forward_hook(capture_activation)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(model_input)
            if not activation or not isinstance(logits, torch.Tensor):
                return {
                    "status": "not_generated",
                    "method": "Grad-CAM",
                    "reason": "The model did not expose tensor activations and logits for explanation.",
                    "target_layer": layer_name,
                }
            class_logits = logits.reshape(logits.shape[0], -1)
            if target_class_index >= class_logits.shape[1]:
                return {
                    "status": "not_generated",
                    "method": "Grad-CAM",
                    "reason": "The selected class logit is not present in the model output.",
                    "target_layer": layer_name,
                }
            target_logit = class_logits[0, target_class_index]
            gradients = torch.autograd.grad(target_logit, activation[-1])[0]
            feature_maps = activation[-1].detach()
            
            # Grad-CAM++ weighting
            grads_power_2 = gradients.pow(2)
            grads_power_3 = grads_power_2 * gradients
            sum_activations = torch.sum(feature_maps, dim=(2, 3), keepdim=True)
            denom = 2 * grads_power_2 + sum_activations * grads_power_3
            denom = torch.where(denom != 0.0, denom, torch.ones_like(denom))
            alphas = grads_power_2 / denom
            
            channel_weights = (alphas * torch.relu(gradients)).sum(dim=(2, 3), keepdim=True)
            cam = torch.relu((channel_weights * feature_maps).sum(dim=1, keepdim=True))
            
            cam = torch.nn.functional.interpolate(
                cam,
                size=(output_size[1], output_size[0]),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            minimum = cam.min()
            maximum = cam.max()
            if float((maximum - minimum).detach().cpu()) > 1e-12:
                cam = (cam - minimum) / (maximum - minimum)
            else:
                cam = torch.zeros_like(cam)
            heat = np.uint8(cam.detach().cpu().numpy() * 255)
    except (RuntimeError, ValueError, IndexError) as error:
        return {
            "status": "not_generated",
            "method": "Grad-CAM",
            "reason": f"The model explanation pass failed: {error}",
            "target_layer": layer_name,
        }
    finally:
        hook.remove()

    image = source_image.convert("RGB").resize(output_size, Image.Resampling.LANCZOS)
    color_heat = cv2.cvtColor(cv2.applyColorMap(heat, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    overlay_pixels = cv2.addWeighted(np.asarray(image), 0.58, color_heat, 0.42, 0)
    output = BytesIO()
    Image.fromarray(overlay_pixels).save(output, format="JPEG", quality=82, optimize=True)
    data_url = "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")
    return {
        "status": "generated",
        "method": "Grad-CAM++",
        "target": "fake-class logit",
        "target_layer": layer_name,
        "applied_to": "model_input_image",
        "overlay_data_url": data_url,
        "overlay_width": output_size[0],
        "overlay_height": output_size[1],
        "interpretation": (
            "This Grad-CAM++ overlay highlights image regions that influenced the model's fake-class "
            "logit. It is a coarse model explanation, not a manipulation mask or proof."
        ),
    }


class XceptionDetector:
    def __init__(self, model: Any, checkpoint: Mapping[str, Any], torch: Any, transforms: Any):
        self.model = model
        self.checkpoint = checkpoint
        self.torch = torch
        self.device = next(model.parameters()).device
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
        ])
        self.face_cascade = _load_face_cascade()

    def predict(self, image_path: str | Path) -> dict[str, Any]:
        try:
            with Image.open(image_path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.load()
        except (OSError, ValueError) as error:
            raise XceptionModelError(f"Xception could not decode the image: {error}") from error

        return self.predict_image(image)

    def predict_frame(self, image: Image.Image) -> dict[str, Any]:
        return self.predict_image(image, include_explainability=False)

    def predict_image(
        self,
        image: Image.Image,
        include_explainability: bool = True,
    ) -> dict[str, Any]:
        image = image.convert("RGB")
        crop, face_review = detect_largest_face_crop(image, self.face_cascade)
        try:
            tensor = self.transform(crop).unsqueeze(0).to(self.device)
            with self.torch.inference_mode():
                logit = self.model(tensor).reshape(-1)[0]
                temperature = self.checkpoint["calibration"]["temperature"]
                fake_probability = float(self.torch.sigmoid(logit / temperature).item())
        except (RuntimeError, ValueError) as error:
            raise XceptionModelError(f"Xception inference failed: {error}") from error
        explainability = (
            _gradcam_explanation(self.model, tensor, crop, self.torch)
            if include_explainability
            else None
        )
        if explainability is not None:
            explainability["model_input_description"] = (
                face_review["selected_crop"].replace("_", " ")
            )

        thresholds = self.checkpoint["thresholds"]
        real_max = thresholds["real_max_probability"]
        fake_min = thresholds["fake_min_probability"]
        if real_max is None:
            decision = "not_determined"
            verdict = "Insufficient evidence"
            status = "calibrated_probability_only"
        elif fake_probability <= real_max:
            decision = "likely_real"
            verdict = "Likely original"
            status = "classified"
        elif fake_probability >= fake_min:
            decision = "likely_fake"
            verdict = "Likely deepfake"
            status = "classified"
        else:
            decision = "suspicious"
            verdict = "Suspicious"
            status = "classified"

        calibration = self.checkpoint["calibration"]
        conformal = calibration.get("conformal_prediction")
        conformal_result = None
        if conformal is not None:
            prediction_set = []
            if fake_probability <= conformal["quantile_real_nonconformity"]:
                prediction_set.append("real")
            if 1.0 - fake_probability <= conformal["quantile_fake_nonconformity"]:
                prediction_set.append("fake")
            expected_class = {
                "likely_real": "real",
                "likely_fake": "fake",
            }.get(decision)
            abstained = len(prediction_set) != 1 or (
                expected_class is not None and prediction_set != [expected_class]
            )
            conformal_result = {
                "prediction_set": prediction_set,
                "alpha": conformal["alpha"],
                "abstained": abstained,
                "scope": conformal["calibration_scope"],
            }
            if expected_class is not None and abstained:
                decision = "not_determined"
                verdict = "Insufficient evidence"
                status = "conformal_abstention"

        return {
            "status": status,
            "decision": decision,
            "verdict": verdict,
            "deepfake_probability": fake_probability,
            "authenticity_probability": 1.0 - fake_probability,
            "model": {
                "architecture": ARCHITECTURE,
                "inference_device": str(self.device),
                "validation_auc": self.checkpoint["validation_auc"],
                "calibration": "held-out temperature scaling",
                "calibration_temperature": calibration["temperature"],
                "calibration_samples_per_class": calibration["samples_per_class"],
                "calibration_metrics_before_scaling": calibration.get("metrics_before_scaling"),
                "calibration_metrics_after_scaling": calibration.get("metrics_after_scaling"),
                "conformal_prediction": conformal_result,
                "thresholds": {
                    "real_max_probability": real_max,
                    "fake_min_probability": fake_min,
                },
                "validation_metrics": self.checkpoint.get("validation_metrics"),
                "validation_metrics_temperature_scaled": self.checkpoint.get(
                    "validation_metrics_temperature_scaled"
                ),
                "test_metrics": self.checkpoint.get("test_metrics"),
                "test_metrics_by_source": self.checkpoint.get("test_metrics_by_source"),
                "dataset_audit": self.checkpoint.get("dataset_audit"),
            },
            "conformal_prediction": conformal_result,
            "face_review": face_review,
            "explainability": explainability,
            "limitations": [
                "Face crops are detected with an OpenCV Haar cascade; this detector can miss faces or return false detections.",
                "Facial landmark alignment is not performed.",
                "The probability reflects this checkpoint's held-out calibration data and may be unreliable under dataset or domain shift.",
                "Grad-CAM is a coarse model-attention explanation, not pixel-level manipulation localization.",
                *(
                    [
                        "Conformal prediction-set coverage assumes future examples are exchangeable with the calibration data; it is not guaranteed under domain shift."
                    ]
                    if conformal_result
                    else []
                ),
                "Grad-CAM is a coarse model-influence visualization, not pixel-level manipulation localization.",
            ],
        }


class PublicXceptionDetector:
    def __init__(self, model: Any, torch: Any, transforms: Any):
        self.model = model
        self.torch = torch
        self.device = next(model.parameters()).device
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.face_cascade = _load_face_cascade()

    def predict(self, image_path: str | Path) -> dict[str, Any]:
        try:
            with Image.open(image_path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.load()
        except (OSError, ValueError) as error:
            raise XceptionModelError(f"Xception could not decode the image: {error}") from error

        return self.predict_image(image)

    def predict_frame(self, image: Image.Image) -> dict[str, Any]:
        return self.predict_image(image, include_explainability=False)

    def predict_image(
        self,
        image: Image.Image,
        include_explainability: bool = True,
    ) -> dict[str, Any]:
        image = image.convert("RGB")
        _, face_review = detect_largest_face_crop(image, self.face_cascade)
        face_review["selected_crop"] = "not_used_full_image_model_input"
        try:
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            with self.torch.inference_mode():
                logits = self.model(tensor)
                probabilities = self.torch.softmax(logits, dim=1)[0]
                fake_probability = float(probabilities[0].item())
                authenticity_probability = float(probabilities[1].item())
        except (RuntimeError, ValueError) as error:
            raise XceptionModelError(f"Xception inference failed: {error}") from error
        explainability = (
            _gradcam_explanation(self.model, tensor, image, self.torch, target_class_index=0)
            if include_explainability
            else None
        )
        if explainability is not None:
            explainability["model_input_description"] = (
                "Full image resized to the model input; no face crop."
            )

        is_fake = fake_probability >= authenticity_probability
        return {
            "status": "uncalibrated_prediction",
            "decision": "likely_fake" if is_fake else "likely_real",
            "verdict": "Likely deepfake" if is_fake else "Likely original",
            "deepfake_probability": fake_probability,
            "authenticity_probability": authenticity_probability,
            "model": {
                "architecture": PUBLIC_ARCHITECTURE,
                "revision": PUBLIC_MODEL_REVISION,
                "checkpoint_sha256": PUBLIC_CHECKPOINT_SHA256,
                "inference_device": str(self.device),
                "license": "MIT",
                "training_data": "140k Real and Fake Faces; real FFHQ and StyleGAN-generated faces",
                "reported_validation_accuracy": "99.36% (model-card claim; not independently verified)",
                "calibration": "uncalibrated raw softmax scores",
                "calibration_temperature": None,
                "thresholds": None,
                "class_mapping": {"fake": 0, "real": 1},
                "input_preprocessing": (
                    "EXIF orientation corrected; full image resized to 299x299 RGB; "
                    "tensor normalized with mean/std [0.5, 0.5, 0.5]"
                ),
            },
            "face_review": {
                **face_review,
                "model_input": "full_image",
            },
            "explainability": explainability,
            "limitations": [
                "Raw softmax scores are uncalibrated and are not validated probabilities of authenticity.",
                "The checkpoint was trained on FFHQ real faces and StyleGAN-generated fake faces; performance on other manipulation or generation methods is unknown.",
                "The model card reports validation accuracy on its own split; that result has not been independently reproduced in this project.",
                "This image classifier is designed for still face images and is not a video, audio, or non-face detector.",
                "Performance may degrade after compression, resizing, screenshots, or other domain shifts.",
                "No held-out calibration set or operating thresholds are provided.",
                "Grad-CAM is a coarse model-attention explanation, not pixel-level manipulation localization.",
            ],
        }


@lru_cache(maxsize=2)
def _load_cached_detector(
    checkpoint_path: str,
    modified_ns: int,
    file_size: int,
) -> XceptionDetector | PublicXceptionDetector:
    del modified_ns, file_size
    try:
        torch = import_module("torch")
        transforms = import_module("torchvision.transforms")
    except ImportError as error:
        raise XceptionModelError(
            "An Xception checkpoint is configured, but PyTorch or torchvision could not "
            f"be imported ({error}). Install requirements-ml.txt."
        ) from error

    checkpoint_file = Path(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=True)
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        raise XceptionModelError(f"Could not load Xception checkpoint: {error}") from error
    if not isinstance(checkpoint, Mapping):
        raise XceptionModelError("Xception checkpoint root must be a metadata object")

    if "architecture" not in checkpoint:
        actual_digest = _sha256_file(checkpoint_file)
        if actual_digest != PUBLIC_CHECKPOINT_SHA256:
            raise XceptionModelError(
                "Unrecognized raw Xception state dictionary; its SHA-256 does not match the "
                "pinned, verified public checkpoint."
            )
        try:
            from models.public_xception import create_model

            model = create_model(torch)
            model.load_state_dict(checkpoint, strict=True)
            model.to(device)
        except (ImportError, RuntimeError, TypeError, ValueError) as error:
            raise XceptionModelError(
                f"Could not initialize the pinned {PUBLIC_ARCHITECTURE} checkpoint: {error}"
            ) from error
        model.eval()
        return PublicXceptionDetector(model, torch, transforms)

    validate_checkpoint_contract(checkpoint)

    try:
        timm = import_module("timm")
        model = timm.create_model("legacy_xception", pretrained=False, num_classes=1)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(device)
    except (ImportError, RuntimeError) as error:
        raise XceptionModelError(
            f"Could not initialize {ARCHITECTURE} or load the checkpoint weights: {error}"
        ) from error
    model.eval()
    return XceptionDetector(model, checkpoint, torch, transforms)


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise XceptionModelError(f"Could not verify Xception checkpoint SHA-256: {error}") from error
    return digest.hexdigest()


def get_xception_detector(
    checkpoint_path: str | Path,
) -> XceptionDetector | PublicXceptionDetector | None:
    path = Path(checkpoint_path)
    if not path.is_file():
        return None
    try:
        file_stat = path.stat()
    except OSError as error:
        raise XceptionModelError(f"Could not inspect Xception checkpoint: {error}") from error
    return _load_cached_detector(str(path.resolve()), file_stat.st_mtime_ns, file_stat.st_size)
