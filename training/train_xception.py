from __future__ import annotations

import argparse
from importlib import import_module
import json
import math
import random
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageOps

from app.services.xception_detector import (
    ARCHITECTURE,
    IMAGE_SIZE,
    LABEL_MAPPING,
    MIN_CALIBRATION_SAMPLES_PER_CLASS,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
    detect_largest_face_crop,
)
from training.dataset_audit import audit_dataset, collect_split_examples, write_audit_report

CONFORMAL_ALPHA = 0.10


def calculate_auc(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    if len(probabilities) != len(labels) or not probabilities:
        raise ValueError("AUC requires equally sized, non-empty probability and label sequences")
    positives = sum(label == 1 for label in labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUC requires at least one example from each class")

    ordered = sorted(zip(probabilities, labels), key=lambda item: item[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2
        positive_rank_sum += average_rank * sum(
            label == 1 for _, label in ordered[index:end]
        )
        index = end
    return (
        positive_rank_sum - positives * (positives + 1) / 2
    ) / (positives * negatives)


def calculate_average_precision(
    probabilities: Sequence[float],
    labels: Sequence[int],
) -> float:
    if len(probabilities) != len(labels) or not probabilities:
        raise ValueError("Average precision requires equally sized, non-empty sequences")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("Average precision labels must be binary: real=0, fake=1")
    if any(not math.isfinite(score) or not 0 <= score <= 1 for score in probabilities):
        raise ValueError("Average precision scores must be finite probabilities")
    positives = sum(labels)
    if positives == 0:
        raise ValueError("Average precision requires at least one fake example")

    previous_recall = 0.0
    average_precision = 0.0
    for threshold in sorted(set(probabilities), reverse=True):
        true_positives = sum(
            score >= threshold and label == 1
            for score, label in zip(probabilities, labels)
        )
        false_positives = sum(
            score >= threshold and label == 0
            for score, label in zip(probabilities, labels)
        )
        recall = true_positives / positives
        precision = true_positives / (true_positives + false_positives)
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
    return average_precision


def calculate_class_conditional_conformal_thresholds(
    probabilities: Sequence[float],
    labels: Sequence[int],
    alpha: float = CONFORMAL_ALPHA,
) -> dict[str, Any]:
    """Calculate split-conformal nonconformity quantiles separately for both labels."""
    if len(probabilities) != len(labels) or not probabilities:
        raise ValueError("Conformal calibration requires equally sized, non-empty sequences")
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("Conformal alpha must be finite and strictly between zero and one")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("Conformal labels must be binary: real=0, fake=1")
    if any(not math.isfinite(score) or not 0 <= score <= 1 for score in probabilities):
        raise ValueError("Conformal scores must be finite probabilities")

    quantiles: dict[str, float] = {}
    counts: dict[str, int] = {}
    for label, name in ((0, "real"), (1, "fake")):
        nonconformity = sorted(
            probability if actual_label == 0 else 1.0 - probability
            for probability, actual_label in zip(probabilities, labels)
            if actual_label == label
        )
        counts[name] = len(nonconformity)
        if not nonconformity:
            raise ValueError("Conformal calibration requires examples from each class")
        rank = math.ceil((len(nonconformity) + 1) * (1.0 - alpha))
        quantiles[name] = nonconformity[min(rank, len(nonconformity)) - 1]

    return {
        "method": "class_conditional_split_conformal",
        "alpha": alpha,
        "quantile_real_nonconformity": quantiles["real"],
        "quantile_fake_nonconformity": quantiles["fake"],
        "samples_per_class": counts,
        "calibration_scope": (
            "Class-conditional split-conformal calibration-set quantiles; coverage interpretation "
            "assumes exchangeability with future examples and is not guaranteed under domain shift."
        ),
    }


def select_operating_thresholds(
    probabilities: Sequence[float],
    labels: Sequence[int],
    minimum_sensitivity_specificity: float,
) -> dict[str, Any]:
    if len(probabilities) != len(labels) or not probabilities:
        raise ValueError("Threshold selection requires equally sized, non-empty sequences")
    if not 0 < minimum_sensitivity_specificity <= 1:
        raise ValueError("The minimum operating point must be greater than zero and at most one")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("Threshold selection labels must be binary: real=0, fake=1")
    if any(not math.isfinite(score) or not 0 <= score <= 1 for score in probabilities):
        raise ValueError("Threshold selection scores must be finite probabilities")

    positives = sum(label == 1 for label in labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("Threshold selection requires examples from both classes")

    candidates = sorted({0.0, 1.0, *probabilities})
    real_options = []
    fake_options = []
    for threshold in candidates:
        real_decisions = [score > threshold for score in probabilities]
        real_sensitivity = sum(
            decision and label == 1 for decision, label in zip(real_decisions, labels)
        ) / positives
        real_specificity = sum(
            not decision and label == 0 for decision, label in zip(real_decisions, labels)
        ) / negatives
        if real_specificity >= minimum_sensitivity_specificity:
            real_options.append((real_sensitivity, threshold, real_specificity))

        fake_decisions = [score >= threshold for score in probabilities]
        fake_sensitivity = sum(
            decision and label == 1 for decision, label in zip(fake_decisions, labels)
        ) / positives
        fake_specificity = sum(
            not decision and label == 0 for decision, label in zip(fake_decisions, labels)
        ) / negatives
        if fake_sensitivity >= minimum_sensitivity_specificity:
            fake_options.append((fake_specificity, threshold, fake_sensitivity))

    real_option = max(real_options, default=None, key=lambda item: (item[0], item[1]))
    fake_option = max(fake_options, default=None, key=lambda item: (item[0], item[1]))
    has_gap = real_option is not None and fake_option is not None and real_option[1] < fake_option[1]
    return {
        "real_max_probability": real_option[1] if has_gap else None,
        "fake_min_probability": fake_option[1] if has_gap else None,
        "status": "available" if has_gap else "no_non_overlapping_operating_gap",
        "minimum_sensitivity_specificity": minimum_sensitivity_specificity,
        "real_threshold_observed_sensitivity": real_option[0] if has_gap else None,
        "real_threshold_observed_specificity": real_option[2] if has_gap else None,
        "fake_threshold_observed_sensitivity": fake_option[2] if has_gap else None,
        "fake_threshold_observed_specificity": fake_option[0] if has_gap else None,
    }


def _build_transforms(training: bool, transforms: Any, torch: Any) -> Any:
    if not training:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
        ])

    class RandomJpegCompression:
        def __call__(self, image: Image.Image) -> Image.Image:
            import io

            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=random.randint(55, 95))
            buffer.seek(0)
            with Image.open(buffer) as compressed:
                return compressed.convert("RGB")

    class RandomGaussianNoise:
        def __call__(self, image_tensor: Any) -> Any:
            return (image_tensor + torch.randn_like(image_tensor) * 0.01).clamp(0, 1)

    return transforms.Compose([
        transforms.RandomResizedCrop(
            IMAGE_SIZE,
            scale=(0.80, 1.0),
            ratio=(0.85, 1.15),
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.15),
        transforms.RandomApply([RandomJpegCompression()], p=0.35),
        transforms.ToTensor(),
        transforms.RandomApply([RandomGaussianNoise()], p=0.20),
        transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
    ])


def _build_dataset(examples: list[tuple[Path, int]], transform: Any, torch: Any) -> Any:
    class ImageEvidenceDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(examples)

        def __getitem__(self, index: int) -> tuple[Any, Any]:
            path, label = examples[index]
            try:
                with Image.open(path) as source:
                    if getattr(source, "is_animated", False):
                        raise ValueError(f"Animated image is not supported for Xception training: {path}")
                    image = ImageOps.exif_transpose(source).convert("RGB")
                    image.load()
            except (OSError, ValueError) as error:
                raise ValueError(f"Could not load training image {path}: {error}") from error
            face_crop, _ = detect_largest_face_crop(image)
            return transform(face_crop), torch.tensor(float(label), dtype=torch.float32)

    return ImageEvidenceDataset()


def _fit_temperature(logits: Sequence[float], labels: Sequence[int], torch: Any, device: Any) -> float:
    logit_tensor = torch.tensor(logits, dtype=torch.float64, device=device)
    label_tensor = torch.tensor(labels, dtype=torch.float64, device=device)
    log_temperature = torch.nn.Parameter(torch.zeros((), dtype=torch.float64, device=device))
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=100,
        line_search_fn="strong_wolfe",
    )

    def closure() -> Any:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logit_tensor / temperature,
            label_tensor,
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0).item())


def _evaluate(model: Any, loader: Any, torch: Any, device: Any) -> tuple[list[float], list[int], float]:
    criterion = torch.nn.BCEWithLogitsLoss()
    logits: list[float] = []
    labels: list[int] = []
    total_loss = 0.0
    model.eval()
    with torch.inference_mode():
        for images, batch_labels in loader:
            images = images.to(device)
            batch_labels = batch_labels.to(device)
            batch_logits = model(images).reshape(-1)
            total_loss += float(criterion(batch_logits, batch_labels).item()) * len(batch_labels)
            logits.extend(float(value) for value in batch_logits.cpu().tolist())
            labels.extend(int(value) for value in batch_labels.cpu().tolist())
    return logits, labels, total_loss / len(labels)


def _probabilities(logits: Sequence[float], temperature: float) -> list[float]:
    return [
        1.0 / (1.0 + math.exp(-max(-80.0, min(80.0, logit / temperature))))
        for logit in logits
    ]


def _classification_metrics(
    probabilities: Sequence[float],
    labels: Sequence[int],
    bin_count: int = 10,
) -> dict[str, Any]:
    if len(probabilities) != len(labels) or not probabilities:
        raise ValueError("Classification metrics require equally sized, non-empty sequences")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("Classification metric labels must be binary: real=0, fake=1")
    if any(not math.isfinite(score) or not 0 <= score <= 1 for score in probabilities):
        raise ValueError("Classification metric scores must be finite probabilities")
    if bin_count < 1:
        raise ValueError("Calibration bin count must be at least one")

    predictions = [int(probability >= 0.5) for probability in probabilities]
    true_positive = sum(prediction == 1 and label == 1 for prediction, label in zip(predictions, labels))
    true_negative = sum(prediction == 0 and label == 0 for prediction, label in zip(predictions, labels))
    false_positive = sum(prediction == 1 and label == 0 for prediction, label in zip(predictions, labels))
    false_negative = sum(prediction == 0 and label == 1 for prediction, label in zip(predictions, labels))
    positive_count = true_positive + false_negative
    negative_count = true_negative + false_positive
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / positive_count if positive_count else 0.0
    specificity = true_negative / negative_count if negative_count else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    calibration_bins = []
    expected_calibration_error = 0.0
    for bin_index in range(bin_count):
        lower = bin_index / bin_count
        upper = (bin_index + 1) / bin_count
        members = [
            (probability, label)
            for probability, label in zip(probabilities, labels)
            if lower <= probability < upper or (bin_index == bin_count - 1 and probability == 1.0)
        ]
        if not members:
            continue
        mean_probability = sum(probability for probability, _ in members) / len(members)
        observed_fake_rate = sum(label for _, label in members) / len(members)
        expected_calibration_error += (
            len(members) / len(labels) * abs(mean_probability - observed_fake_rate)
        )
        calibration_bins.append({
            "lower_bound": lower,
            "upper_bound": upper,
            "samples": len(members),
            "mean_predicted_fake_rate": mean_probability,
            "observed_fake_rate": observed_fake_rate,
        })

    return {
        "samples": len(labels),
        "samples_per_class": {
            "real": negative_count,
            "fake": positive_count,
        },
        "accuracy_at_0_5": sum(
            prediction == label for prediction, label in zip(predictions, labels)
        ) / len(labels),
        "balanced_accuracy_at_0_5": (recall + specificity) / 2,
        "precision_at_0_5": precision,
        "recall_at_0_5": recall,
        "specificity_at_0_5": specificity,
        "f1_at_0_5": f1_score,
        "brier_score": sum(
            (probability - label) ** 2
            for probability, label in zip(probabilities, labels)
        ) / len(labels),
        "expected_calibration_error": expected_calibration_error,
        "calibration_bins": calibration_bins,
        "confusion_matrix_real_fake": [
            [true_negative, false_positive],
            [false_negative, true_positive],
        ],
    }


def _binary_cross_entropy(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    if len(probabilities) != len(labels) or not probabilities:
        raise ValueError("Binary cross entropy requires equally sized, non-empty sequences")
    epsilon = 1e-15
    losses = []
    for probability, label in zip(probabilities, labels):
        bounded_probability = min(1.0 - epsilon, max(epsilon, probability))
        losses.append(
            -(
                label * math.log(bounded_probability)
                + (1 - label) * math.log(1.0 - bounded_probability)
            )
        )
    return sum(losses) / len(losses)


def _operating_point_metrics(
    probabilities: Sequence[float],
    labels: Sequence[int],
    real_max_probability: float | None,
    fake_min_probability: float | None,
) -> dict[str, Any]:
    if real_max_probability is None or fake_min_probability is None:
        return {"status": "not_available"}
    real_decisions = [
        "real" if probability <= real_max_probability
        else "fake" if probability >= fake_min_probability
        else "abstain"
        for probability in probabilities
    ]
    decided = [
        (decision, label)
        for decision, label in zip(real_decisions, labels)
        if decision != "abstain"
    ]
    correct = sum(
        (decision == "real" and label == 0) or (decision == "fake" and label == 1)
        for decision, label in decided
    )
    return {
        "status": "available",
        "samples": len(labels),
        "decided_samples": len(decided),
        "abstained_samples": len(labels) - len(decided),
        "coverage": len(decided) / len(labels),
        "accuracy_on_decided_samples": correct / len(decided) if decided else None,
        "real_decisions": real_decisions.count("real"),
        "fake_decisions": real_decisions.count("fake"),
        "real_threshold": real_max_probability,
        "fake_threshold": fake_min_probability,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    try:
        timm = import_module("timm")
        torch = import_module("torch")
        DataLoader = import_module("torch.utils.data").DataLoader
        transforms = import_module("torchvision.transforms")
    except ImportError as error:
        raise RuntimeError(
            "Training requires PyTorch, torchvision and timm. Install requirements-ml.txt."
        ) from error

    data_root = Path(args.data_root).resolve()
    if not (data_root / "test").is_dir():
        raise ValueError(
            "An independent test/ split is required for final benchmark reporting; "
            "it must not be used for training, model selection, calibration, or threshold selection."
        )
    split_names = ["train", "validation", "calibration", "test"]
    split_examples = collect_split_examples(data_root, split_names)
    split_counts = {
        split: {
            label: sum(example_label == class_id for _, example_label in examples)
            for label, class_id in LABEL_MAPPING.items()
        }
        for split, examples in split_examples.items()
    }
    for split in ("train", "validation"):
        if any(count == 0 for count in split_counts[split].values()):
            raise ValueError(f"Both real and fake examples are required in the {split} split")
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
    group_manifest_argument = getattr(args, "group_manifest", None)
    if not group_manifest_argument:
        raise ValueError(
            "Training requires --group-manifest with source/person group IDs, source, and license "
            "for every image so cross-split group leakage can be checked."
        )
    candidate = Path(group_manifest_argument)
    group_manifest_path = candidate if candidate.is_absolute() else data_root / candidate
    dataset_audit = audit_dataset(data_root, split_examples, group_manifest_path)
    output_path = Path(args.output).resolve()
    audit_report_argument = getattr(args, "audit_report", None)
    audit_report_path = (
        Path(audit_report_argument).resolve()
        if audit_report_argument
        else output_path.with_name(f"{output_path.stem}.dataset-audit.json")
    )
    if audit_report_path == output_path:
        raise ValueError("Dataset audit report and model checkpoint must use different output paths")
    write_audit_report(dataset_audit, audit_report_path)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets = {
        split: _build_dataset(
            examples,
            _build_transforms(split == "train", transforms, torch),
            torch,
        )
        for split, examples in split_examples.items()
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )
        for split, dataset in datasets.items()
    }

    model = timm.create_model(
        "legacy_xception",
        pretrained=not args.no_pretrained,
        num_classes=1,
    ).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_auc = -1.0
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    epochs_without_improvement = 0
    best_validation_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_train_examples = 0
        for images, labels in loaders["train"]:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images).reshape(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += float(loss.item()) * len(labels)
            total_train_examples += len(labels)

        validation_logits, validation_labels, validation_loss = _evaluate(
            model,
            loaders["validation"],
            torch,
            device,
        )
        validation_probabilities = _probabilities(validation_logits, 1.0)
        validation_auc = calculate_auc(validation_probabilities, validation_labels)
        print(json.dumps({
            "epoch": epoch,
            "training_loss": total_train_loss / total_train_examples,
            "validation_loss": validation_loss,
            "validation_auc": validation_auc,
        }))

        if validation_auc > best_auc or (
            math.isclose(validation_auc, best_auc) and validation_loss < best_validation_loss
        ):
            best_auc = validation_auc
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("Training completed without a finite best-validation checkpoint")
    if best_auc <= 0.5:
        raise ValueError(
            f"Best validation AUC was {best_auc:.4f}; refuse to export a detector at or below chance"
        )
    model.load_state_dict(best_state, strict=True)
    model.to(device)

    validation_logits, validation_labels, validation_loss = _evaluate(
        model,
        loaders["validation"],
        torch,
        device,
    )
    validation_probabilities = _probabilities(validation_logits, 1.0)
    validation_auc = calculate_auc(validation_probabilities, validation_labels)
    validation_metrics = _classification_metrics(validation_probabilities, validation_labels)
    validation_metrics["roc_auc"] = validation_auc
    validation_metrics["pr_auc"] = calculate_average_precision(
        validation_probabilities,
        validation_labels,
    )
    validation_metrics["binary_cross_entropy_loss"] = validation_loss
    calibration_logits, calibration_labels, calibration_loss = _evaluate(
        model,
        loaders["calibration"],
        torch,
        device,
    )
    temperature = _fit_temperature(calibration_logits, calibration_labels, torch, device)
    calibration_probabilities_before_scaling = _probabilities(calibration_logits, 1.0)
    calibration_metrics_before_scaling = _classification_metrics(
        calibration_probabilities_before_scaling,
        calibration_labels,
    )
    calibration_probabilities = _probabilities(calibration_logits, temperature)
    calibration_metrics_after_scaling = _classification_metrics(
        calibration_probabilities,
        calibration_labels,
    )
    conformal_thresholds = calculate_class_conditional_conformal_thresholds(
        calibration_probabilities,
        calibration_labels,
    )
    thresholds = select_operating_thresholds(
        calibration_probabilities,
        calibration_labels,
        args.minimum_operating_point,
    )
    calibration_operating_metrics = _operating_point_metrics(
        calibration_probabilities,
        calibration_labels,
        thresholds["real_max_probability"],
        thresholds["fake_min_probability"],
    )
    validation_probabilities_temperature_scaled = _probabilities(validation_logits, temperature)
    validation_metrics_temperature_scaled = _classification_metrics(
        validation_probabilities_temperature_scaled,
        validation_labels,
    )
    validation_metrics_temperature_scaled["roc_auc"] = calculate_auc(
        validation_probabilities_temperature_scaled,
        validation_labels,
    )
    validation_metrics_temperature_scaled["pr_auc"] = calculate_average_precision(
        validation_probabilities_temperature_scaled,
        validation_labels,
    )
    validation_metrics_temperature_scaled["binary_cross_entropy_loss"] = _binary_cross_entropy(
        validation_probabilities_temperature_scaled,
        validation_labels,
    )

    test_metrics = None
    test_metrics_by_source = None
    if "test" in loaders:
        test_logits, test_labels, _ = _evaluate(
            model,
            loaders["test"],
            torch,
            device,
        )
        test_probabilities = _probabilities(test_logits, temperature)
        test_metrics = _classification_metrics(test_probabilities, test_labels)
        test_metrics["roc_auc"] = calculate_auc(test_probabilities, test_labels)
        test_metrics["pr_auc"] = calculate_average_precision(test_probabilities, test_labels)
        test_metrics["binary_cross_entropy_loss"] = _binary_cross_entropy(
            test_probabilities,
            test_labels,
        )
        test_metrics["evaluation_scope"] = (
            "Independent test split; not used for training, model selection, calibration, or threshold selection."
        )
        test_metrics["source_group_independence"] = dataset_audit["group_overlap_status"]
        audit_files_by_path = {
            item["path"]: item
            for item in dataset_audit["files"]
            if item["split"] == "test"
        }
        source_examples: dict[str, tuple[list[float], list[int]]] = {}
        for (path, label), probability in zip(
            split_examples["test"],
            test_probabilities,
        ):
            relative_path = path.resolve(strict=True).relative_to(data_root).as_posix()
            source = audit_files_by_path[relative_path]["source"]
            source_probabilities, source_labels = source_examples.setdefault(
                source,
                ([], []),
            )
            source_probabilities.append(probability)
            source_labels.append(label)
        source_split_summary = dataset_audit["source_split_summary"]
        test_metrics_by_source = {}
        for source, (source_probabilities, source_labels) in sorted(source_examples.items()):
            source_metrics = _classification_metrics(source_probabilities, source_labels)
            source_metrics["roc_auc"] = (
                calculate_auc(source_probabilities, source_labels)
                if len(set(source_labels)) == 2
                else None
            )
            source_metrics["pr_auc"] = (
                calculate_average_precision(source_probabilities, source_labels)
                if 1 in source_labels
                else None
            )
            source_metrics["calibration_operating_point"] = _operating_point_metrics(
                source_probabilities,
                source_labels,
                thresholds["real_max_probability"],
                thresholds["fake_min_probability"],
            )
            source_metrics["source_not_seen_in_train_validation_or_calibration"] = not any(
                split_name in {"train", "validation", "calibration"}
                for split_name in source_split_summary.get(source, [])
            )
            test_metrics_by_source[source] = source_metrics
        test_metrics["calibration_operating_point"] = _operating_point_metrics(
            test_probabilities,
            test_labels,
            thresholds["real_max_probability"],
            thresholds["fake_min_probability"],
        )
    thresholds_for_checkpoint = {
        "real_max_probability": thresholds["real_max_probability"],
        "fake_min_probability": thresholds["fake_min_probability"],
        "calibration_operating_point": thresholds["minimum_sensitivity_specificity"],
        "status": thresholds["status"],
        "observed": {
            "real_threshold_sensitivity": thresholds["real_threshold_observed_sensitivity"],
            "real_threshold_specificity": thresholds["real_threshold_observed_specificity"],
            "fake_threshold_sensitivity": thresholds["fake_threshold_observed_sensitivity"],
            "fake_threshold_specificity": thresholds["fake_threshold_observed_specificity"],
        },
    }

    checkpoint = {
        "architecture": ARCHITECTURE,
        "pretrained_initialization": "ImageNet" if not args.no_pretrained else "disabled_by_argument",
        "image_size": IMAGE_SIZE,
        "num_logits": 1,
        "label_mapping": LABEL_MAPPING,
        "normalization": {
            "mean": list(NORMALIZATION_MEAN),
            "std": list(NORMALIZATION_STD),
        },
        "model_state_dict": best_state,
        "best_epoch": best_epoch,
        "validation_auc": validation_auc,
        "dataset_audit": {
            "manifest_sha256": dataset_audit["manifest_sha256"],
            "group_manifest_sha256": dataset_audit["group_manifest_sha256"],
            "group_overlap_status": dataset_audit["group_overlap_status"],
            "source_split_summary": dataset_audit["source_split_summary"],
            "split_counts": dataset_audit["split_counts"],
            "warnings": dataset_audit["warnings"],
            "audit_report": audit_report_path.name,
        },
        "validation_metrics": validation_metrics,
        "validation_metrics_temperature_scaled": validation_metrics_temperature_scaled,
        "test_metrics": test_metrics,
        "test_metrics_by_source": test_metrics_by_source,
        "calibration": {
            "status": "temperature_scaled",
            "temperature": temperature,
            "method": "single scalar optimized with held-out BCE",
            "evaluation_scope": (
                "Calibration split; used to fit temperature and select thresholds, "
                "not an independent generalization estimate."
            ),
            "samples": len(calibration_labels),
            "samples_per_class": split_counts["calibration"],
            "binary_cross_entropy_loss_before_scaling": calibration_loss,
            "binary_cross_entropy_loss_after_scaling": _binary_cross_entropy(
                calibration_probabilities,
                calibration_labels,
            ),
            "metrics_before_scaling": calibration_metrics_before_scaling,
            "metrics_after_scaling": calibration_metrics_after_scaling,
            "operating_threshold_metrics": calibration_operating_metrics,
            "conformal_prediction": conformal_thresholds,
        },
        "thresholds": thresholds_for_checkpoint,
        "training": {
            "seed": args.seed,
            "epochs_completed": epoch,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "augmentation": [
                "random resized crop",
                "horizontal flip",
                "mild color jitter",
                "Gaussian blur",
                "JPEG compression",
                "Gaussian pixel noise",
            ],
            "image_hash_overlap_check": "none detected across train, validation, calibration, and any supplied test split",
            "source_group_overlap_check": dataset_audit["group_overlap_status"],
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(output_path)
    except OSError:
        temporary_path.unlink(missing_ok=True)
        raise

    result = {
        "checkpoint": str(output_path),
        "best_epoch": best_epoch,
        "validation_auc": validation_auc,
        "dataset_audit_path": str(audit_report_path),
        "dataset_audit_sha256": dataset_audit["manifest_sha256"],
        "group_overlap_status": dataset_audit["group_overlap_status"],
        "test_status": "evaluated" if test_metrics is not None else "not_provided",
        "validation_metrics": validation_metrics,
        "validation_metrics_temperature_scaled": validation_metrics_temperature_scaled,
        "test_metrics": test_metrics,
        "test_metrics_by_source": test_metrics_by_source,
        "calibration_temperature": temperature,
        "calibration_thresholds": thresholds_for_checkpoint,
        "calibration_metrics_before_scaling": calibration_metrics_before_scaling,
        "calibration_metrics_after_scaling": calibration_metrics_after_scaling,
        "calibration_operating_threshold_metrics": calibration_operating_metrics,
        "calibration_conformal_prediction": conformal_thresholds,
    }
    print(json.dumps(result, indent=2))
    return result


def _parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Train and calibrate the DeepTrace static-image Xception detector."
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Directory containing train/, validation/, calibration/, and test/ splits",
    )
    parser.add_argument(
        "--group-manifest",
        required=True,
        help="CSV with path, group_id, source, and license columns for group-leakage checks",
    )
    parser.add_argument(
        "--audit-report",
        help="Dataset audit JSON path (defaults beside the model checkpoint)",
    )
    parser.add_argument(
        "--output",
        default=str(project_root / "backend" / "weights" / "xception_deepfake.pth"),
        help="Output checkpoint path",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--minimum-operating-point",
        type=float,
        required=True,
        help="Minimum calibration sensitivity/specificity for threshold selection (e.g. 0.90)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not initialize from ImageNet weights; not recommended for research-grade training.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(_parse_args())
