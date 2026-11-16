from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

try:
    from modules.data_preprocessor import (
        PROCESSED_TEXT_COLUMN,
        TARGET_COLUMN,
    )
    from modules.feature_extractor import FeatureExtractor
    from modules.trainer import FinalTrainingResult, TuningResult
except ModuleNotFoundError:  # Support direct imports from modules/.
    from data_preprocessor import (  # type: ignore[no-redef]
        PROCESSED_TEXT_COLUMN,
        TARGET_COLUMN,
    )
    from feature_extractor import FeatureExtractor  # type: ignore[no-redef]
    from trainer import (  # type: ignore[no-redef]
        FinalTrainingResult,
        TuningResult,
    )


@dataclass
class ValidationResult:
    selected_model: str
    metrics: pd.DataFrame
    predictions: dict[str, Any]


def _metrics(labels: pd.Series, predictions: Any) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
    }


def _save_test_metrics(
    metrics: pd.DataFrame,
    project_root: Path,
) -> None:
    destination = project_root / "outputs" / "generated" / "test_metrics.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = pd.read_csv(destination)
        metrics = pd.concat([existing, metrics], ignore_index=True)
        metrics = metrics.drop_duplicates(
            subset=["feature_method", "model"],
            keep="last",
        )
    metrics.to_csv(destination, index=False)


def validate_models(
    tuning: TuningResult,
    validation: pd.DataFrame,
    validation_features: Any | None = None,
) -> ValidationResult:
    labels = validation[TARGET_COLUMN]
    metrics: list[dict[str, Any]] = []
    predictions: dict[str, Any] = {}
    if tuning.uses_precomputed_features:
        if validation_features is None:
            raise ValueError("Validation requires reloaded precomputed features")
        if len(validation_features) != len(validation):
            raise ValueError("Precomputed validation features are not row-aligned")
        validation_input = validation_features
    else:
        if validation_features is not None:
            raise ValueError("Unexpected precomputed validation features")
        validation_input = validation[PROCESSED_TEXT_COLUMN]

    for model_name, search in tuning.searches.items():
        prediction = search.best_estimator_.predict(validation_input)
        predictions[model_name] = prediction
        metrics.append(
            {"model": model_name, **_metrics(labels, prediction)}
        )

    metrics_table = pd.DataFrame(metrics)
    selected_model = str(
        metrics_table.sort_values(
            ["macro_f1", "accuracy", "model"],
            ascending=[False, False, True],
        ).iloc[0]["model"]
    )
    return ValidationResult(selected_model, metrics_table, predictions)


def evaluate_models(
    training: FinalTrainingResult,
    feature_extractor: FeatureExtractor,
    test: pd.DataFrame,
    validation: ValidationResult,
    project_root: Path | str = Path("."),
    test_features: Any | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    test_labels = test[TARGET_COLUMN]
    test_metrics: list[dict[str, Any]] = []
    test_predictions: dict[str, Any] = {}
    if training.uses_precomputed_features:
        if test_features is None:
            raise ValueError("Evaluation requires reloaded precomputed features")
        if len(test_features) != len(test):
            raise ValueError("Precomputed test features are not row-aligned")
        test_input = test_features
    else:
        if test_features is not None:
            raise ValueError("Unexpected precomputed test features")
        test_input = test[PROCESSED_TEXT_COLUMN]

    for model_name, pipeline in training.pipelines.items():
        test_prediction = pipeline.predict(test_input)
        test_predictions[model_name] = test_prediction
        test_metrics.append(
            {
                "feature_method": feature_extractor.label,
                "model": model_name,
                **_metrics(test_labels, test_prediction),
            }
        )

    test_table = pd.DataFrame(test_metrics)
    _save_test_metrics(test_table, root)

    return {
        "feature_name": feature_extractor.name,
        "feature_label": feature_extractor.label,
        "selected_model": validation.selected_model,
        "label_order": sorted(test_labels.unique().tolist()),
        "validation_metrics": validation.metrics,
        "test_metrics": test_table,
        "metrics_summary": pd.read_csv(
            root / "outputs" / "generated" / "test_metrics.csv"
        ),
        "validation_predictions": validation.predictions,
        "test_predictions": test_predictions,
        "selected_test_predictions": test_predictions[validation.selected_model],
    }
