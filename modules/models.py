from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_is_fitted

try:
    from modules.data_preprocessor import RANDOM_SEED
except ModuleNotFoundError:  # Support direct imports from modules/.
    from data_preprocessor import RANDOM_SEED  # type: ignore[no-redef]


class ClassificationModel(ClassifierMixin, BaseEstimator, ABC):
    name: str
    c_values: tuple[float, ...]
    C: float
    class_weight: str | None

    @abstractmethod
    def _create_estimator(self) -> BaseEstimator:
        pass

    def fit(self, features: Any, labels: Any) -> ClassificationModel:
        self.estimator_ = self._create_estimator()
        self.estimator_.fit(features, labels)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, features: Any) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(features)

    def parameter_grid(self) -> dict[str, list[Any]]:
        return {
            "model__C": list(self.c_values),
            "model__class_weight": [None, "balanced"],
        }


class LogisticRegressionModel(ClassificationModel):
    name = "logistic_regression"

    def __init__(
        self,
        c_values: tuple[float, ...] = (0.5, 1.0),
        C: float = 1.0,
        class_weight: str | None = None,
    ) -> None:
        self.c_values = c_values
        self.C = C
        self.class_weight = class_weight

    def _create_estimator(self) -> LogisticRegression:
        return LogisticRegression(
            C=self.C,
            class_weight=self.class_weight,
            solver="lbfgs",
            max_iter=2_000,
            random_state=RANDOM_SEED,
        )


class LinearSVMModel(ClassificationModel):
    name = "linear_svm"

    def __init__(
        self,
        c_values: tuple[float, ...] = (0.5, 1.0, 2.0),
        C: float = 1.0,
        class_weight: str | None = None,
    ) -> None:
        self.c_values = c_values
        self.C = C
        self.class_weight = class_weight

    def _create_estimator(self) -> LinearSVC:
        return LinearSVC(
            C=self.C,
            class_weight=self.class_weight,
            random_state=RANDOM_SEED,
        )
