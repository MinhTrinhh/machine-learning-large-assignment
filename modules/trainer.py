from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

try:
    from modules.data_preprocessor import (
        PROCESSED_TEXT_COLUMN,
        RANDOM_SEED,
        TARGET_COLUMN,
    )
    from modules.feature_extractor import FeatureExtractor
    from modules.models import ClassificationModel
except ModuleNotFoundError:  # Support direct imports from modules/.
    from data_preprocessor import (  # type: ignore[no-redef]
        PROCESSED_TEXT_COLUMN,
        RANDOM_SEED,
        TARGET_COLUMN,
    )
    from feature_extractor import FeatureExtractor  # type: ignore[no-redef]
    from models import ClassificationModel  # type: ignore[no-redef]


@dataclass
class PreparedData:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    audit: dict[str, int]

    @property
    def full_train(self) -> pd.DataFrame:
        return pd.concat([self.train, self.validation], ignore_index=True)


@dataclass
class TuningResult:
    feature_extractor: FeatureExtractor
    searches: dict[str, GridSearchCV]
    selected_parameters: dict[str, dict[str, Any]]
    tuning_seconds: dict[str, float]
    uses_precomputed_features: bool


@dataclass
class FinalTrainingResult:
    pipelines: dict[str, Pipeline]
    fit_seconds: dict[str, float]
    uses_precomputed_features: bool


def save_precomputed_feature_representations(
    feature_extractor: FeatureExtractor,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    full_train: pd.DataFrame,
    test: pd.DataFrame,
    project_root: Path | str = Path("."),
) -> dict[str, Any]:
    if feature_extractor.name != "glove":
        raise ValueError(
            "Precomputed .npy classifier inputs are currently supported for GloVe only"
        )

    fitted_extractor = clone(feature_extractor).fit(
        full_train[PROCESSED_TEXT_COLUMN]
    )
    frames = {
        "development": train,
        "validation": validation,
        "train": full_train,
        "test": test,
    }
    features = {
        split: fitted_extractor.transform(frame[PROCESSED_TEXT_COLUMN])
        for split, frame in frames.items()
    }

    destination = Path(project_root).resolve() / "features"
    destination.mkdir(parents=True, exist_ok=True)
    feature_files: dict[str, str] = {}
    for split, matrix in features.items():
        filename = f"{fitted_extractor.file_stem}_{split}.npy"
        np.save(destination / filename, matrix)
        feature_files[split] = filename

    label_files = {
        "development": f"{fitted_extractor.file_stem}_development_labels.npy",
        "validation": f"{fitted_extractor.file_stem}_validation_labels.npy",
        "train": "train_labels.npy",
        "test": "test_labels.npy",
    }
    for split, filename in label_files.items():
        np.save(
            destination / filename,
            frames[split][TARGET_COLUMN].to_numpy(dtype=str),
        )

    metadata = {
        "feature_method": fitted_extractor.label,
        "saved_before_model_training": True,
        "classifier_input": "feature matrices reloaded from saved .npy files",
        "feature_files": feature_files,
        "label_files_by_split": label_files,
        "development_shape": list(features["development"].shape),
        "validation_shape": list(features["validation"].shape),
        "train_shape": list(features["train"].shape),
        "test_shape": list(features["test"].shape),
        "train_rows": len(full_train),
        "test_rows": len(test),
        "train_order": "prepared training partition followed by validation partition",
        "test_order": "official test-file order after preprocessing",
        "classes": sorted(full_train[TARGET_COLUMN].unique().tolist()),
    }
    metadata_path = destination / f"{fitted_extractor.file_stem}_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    return metadata


def load_precomputed_feature_representations(
    feature_extractor: FeatureExtractor,
    project_root: Path | str = Path("."),
) -> dict[str, np.ndarray]:
    """Reload the saved arrays that classifiers must consume."""
    destination = Path(project_root).resolve() / "features"
    metadata_path = destination / f"{feature_extractor.file_stem}_metadata.json"
    with metadata_path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)

    if not metadata.get("saved_before_model_training"):
        raise ValueError("Feature metadata does not confirm pre-training serialization")
    feature_files = metadata.get("feature_files", {})
    expected_splits = {"development", "validation", "train", "test"}
    if set(feature_files) != expected_splits:
        raise ValueError(
            f"Expected saved feature splits {sorted(expected_splits)}; "
            f"found {sorted(feature_files)}"
        )
    return {
        split: np.load(destination / filename, allow_pickle=False)
        for split, filename in feature_files.items()
    }


def save_feature_representations(
    training: FinalTrainingResult,
    full_train: pd.DataFrame,
    test: pd.DataFrame,
    model_name: str,
    project_root: Path | str = Path("."),
) -> dict[str, Any]:
    if model_name not in training.pipelines:
        raise ValueError(f"No fitted pipeline found for model: {model_name}")

    pipeline = training.pipelines[model_name]
    fitted_extractor = pipeline.named_steps["feature_extractor"]
    train_features = fitted_extractor.transform(
        full_train[PROCESSED_TEXT_COLUMN]
    )
    test_features = fitted_extractor.transform(test[PROCESSED_TEXT_COLUMN])

    destination = Path(project_root).resolve() / "features"
    destination.mkdir(parents=True, exist_ok=True)
    fitted_extractor.save_features(train_features, test_features, destination)

    np.save(
        destination / "train_labels.npy",
        full_train[TARGET_COLUMN].to_numpy(dtype=str),
    )
    np.save(
        destination / "test_labels.npy",
        test[TARGET_COLUMN].to_numpy(dtype=str),
    )

    metadata = {
        "feature_method": fitted_extractor.label,
        "validation_selected_model": model_name,
        "train_shape": list(train_features.shape),
        "test_shape": list(test_features.shape),
        "train_rows": len(full_train),
        "test_rows": len(test),
        "train_order": "prepared training partition followed by validation partition",
        "test_order": "official test-file order after preprocessing",
        "label_files": ["train_labels.npy", "test_labels.npy"],
        "classes": sorted(full_train[TARGET_COLUMN].unique().tolist()),
    }
    metadata_path = destination / f"{fitted_extractor.file_stem}_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    return metadata


def _build_search(
    feature_extractor: FeatureExtractor,
    model: ClassificationModel,
    cache_directory: Path,
    use_precomputed_features: bool = False,
) -> GridSearchCV:
    memory = (
        joblib.Memory(cache_directory, verbose=0)
        if feature_extractor.cache_features
        else None
    )
    extractor_steps = [
        (name, clone(transformer))
        for name, transformer in feature_extractor.pipeline_steps()
    ]
    if use_precomputed_features:
        extractor_steps = [
            (name, transformer)
            for name, transformer in extractor_steps
            if name != "feature_extractor"
        ]
        if feature_extractor.parameter_grid():
            raise ValueError(
                "Precomputed features cannot tune feature-extractor parameters"
            )
    steps = extractor_steps
    steps.append(("model", clone(model)))
    parameters = {
        **({} if use_precomputed_features else feature_extractor.parameter_grid()),
        **model.parameter_grid(),
    }
    return GridSearchCV(
        Pipeline(steps, memory=memory),
        parameters,
        scoring={"macro_f1": "f1_macro", "accuracy": "accuracy"},
        refit="macro_f1",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED),
        n_jobs=feature_extractor.search_n_jobs,
        return_train_score=False,
        verbose=1,
    )


def _serialisable_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        name: list(value) if isinstance(value, tuple) else value
        for name, value in parameters.items()
    }


def _candidate_results(search: GridSearchCV) -> pd.DataFrame:
    results = pd.DataFrame(
        {
            "candidate": np.arange(1, len(search.cv_results_["params"]) + 1),
            "parameters": [
                json.dumps(_serialisable_parameters(parameters), sort_keys=True)
                for parameters in search.cv_results_["params"]
            ],
            "mean_cv_macro_f1": search.cv_results_["mean_test_macro_f1"],
            "std_cv_macro_f1": search.cv_results_["std_test_macro_f1"],
            "mean_cv_accuracy": search.cv_results_["mean_test_accuracy"],
            "rank_cv_macro_f1": search.cv_results_["rank_test_macro_f1"],
        }
    )
    results["selected"] = False
    results.loc[int(search.best_index_), "selected"] = True
    return results.sort_values(
        ["rank_cv_macro_f1", "candidate"], ignore_index=True
    )


def _print_and_save_candidate_results(
    search: GridSearchCV,
    feature_extractor: FeatureExtractor,
    model: ClassificationModel,
    project_root: Path,
) -> None:
    results = _candidate_results(search)
    destination = project_root / "outputs" / "generated" / "cv_results"
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / f"{feature_extractor.name}_{model.name}.csv"
    results.to_csv(output_path, index=False)

    display_results = results.copy()
    for column in (
        "mean_cv_macro_f1",
        "std_cv_macro_f1",
        "mean_cv_accuracy",
    ):
        display_results[column] = display_results[column].map(
            lambda value: f"{value:.4f}"
        )

    print(
        f"\nCV results for every {feature_extractor.label} + {model.name} "
        "candidate (sorted by macro-F1 rank):"
    )
    print(display_results.to_string(index=False))
    print(f"Saved candidate results to: {output_path}")


def tune_models(
    feature_extractor: FeatureExtractor,
    models: list[ClassificationModel],
    train: pd.DataFrame,
    project_root: Path | str = Path("."),
    train_features: Any | None = None,
) -> TuningResult:
    root = Path(project_root).resolve()
    cache_directory = (
        root
        / ".cache"
        / "sklearn"
        / feature_extractor.name
    )
    cache_directory.mkdir(parents=True, exist_ok=True)
    searches: dict[str, GridSearchCV] = {}
    selected_parameters: dict[str, dict[str, Any]] = {}
    tuning_seconds: dict[str, float] = {}
    uses_precomputed_features = train_features is not None
    if uses_precomputed_features and len(train_features) != len(train):
        raise ValueError("Precomputed development features are not row-aligned")
    training_input = (
        train_features
        if uses_precomputed_features
        else train[PROCESSED_TEXT_COLUMN]
    )

    for model in models:
        search = _build_search(
            feature_extractor,
            model,
            cache_directory,
            use_precomputed_features=uses_precomputed_features,
        )
        print(f"\nTuning {model.name} on {feature_extractor.label} features ...")
        started = time.perf_counter()
        search.fit(training_input, train[TARGET_COLUMN])
        tuning_seconds[model.name] = time.perf_counter() - started
        _print_and_save_candidate_results(
            search, feature_extractor, model, root
        )
        searches[model.name] = search
        selected_parameters[model.name] = _serialisable_parameters(
            search.best_params_
        )

    return TuningResult(
        feature_extractor,
        searches,
        selected_parameters,
        tuning_seconds,
        uses_precomputed_features,
    )


def fit_final_models(
    tuning: TuningResult,
    full_train: pd.DataFrame,
    project_root: Path | str = Path("."),
    full_train_features: Any | None = None,
) -> FinalTrainingResult:
    model_directory = (
        Path(project_root).resolve()
        / "outputs"
        / "models"
        / tuning.feature_extractor.name
    )
    model_directory.mkdir(parents=True, exist_ok=True)
    pipelines: dict[str, Pipeline] = {}
    fit_seconds: dict[str, float] = {}
    if tuning.uses_precomputed_features:
        if full_train_features is None:
            raise ValueError("Final fitting requires reloaded precomputed features")
        if len(full_train_features) != len(full_train):
            raise ValueError("Precomputed full-training features are not row-aligned")
        training_input = full_train_features
    else:
        if full_train_features is not None:
            raise ValueError("Unexpected precomputed features for a text-input pipeline")
        training_input = full_train[PROCESSED_TEXT_COLUMN]

    for model_name, search in tuning.searches.items():
        pipeline = clone(search.best_estimator_)
        started = time.perf_counter()
        pipeline.fit(training_input, full_train[TARGET_COLUMN])
        fit_seconds[model_name] = time.perf_counter() - started
        joblib.dump(
            pipeline,
            model_directory / f"{model_name}_{tuning.feature_extractor.name}.joblib",
        )
        pipelines[model_name] = pipeline

    return FinalTrainingResult(
        pipelines,
        fit_seconds,
        tuning.uses_precomputed_features,
    )
