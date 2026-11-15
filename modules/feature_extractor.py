from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


GLOVE_MODEL_NAME = "glove-twitter-100"
GLOVE_TOKEN_PATTERN = re.compile(r"<url>|<user>|[a-z0-9_]+(?:'[a-z]+)?|[!?]+")


class FeatureExtractor(BaseEstimator, TransformerMixin, ABC):
    name: str
    label: str
    file_stem: str
    cache_features: bool = False
    search_n_jobs: int = -1

    def pipeline_steps(self) -> list[tuple[str, Any]]:
        return [("feature_extractor", self)]

    def parameter_grid(self) -> dict[str, list[Any]]:
        return {}

    @abstractmethod
    def fit(self, texts: pd.Series, y: Any = None) -> FeatureExtractor:
        pass

    @abstractmethod
    def transform(self, texts: pd.Series) -> Any:
        pass

    @abstractmethod
    def save_features(
        self,
        train_features: Any,
        test_features: Any,
        destination: Path,
    ) -> None:
        pass

class TfidfFeatureExtractor(FeatureExtractor):
    name = "tfidf"
    label = "TF-IDF"
    file_stem = "tfidf"

    def __init__(self, ngram_range: tuple[int, int] = (1, 1)) -> None:
        self.ngram_range = ngram_range

    def _create_vectorizer(self) -> TfidfVectorizer:
        return TfidfVectorizer(
            lowercase=False,
            token_pattern=r"(?u)\b\w[\w']+\b",
            ngram_range=self.ngram_range,
            max_df=0.95,
            min_df=2,
            max_features=50_000,
            sublinear_tf=True,
            dtype=np.float32,
        )

    def fit(self, texts: pd.Series, y: Any = None) -> TfidfFeatureExtractor:
        self.vectorizer_ = self._create_vectorizer().fit(texts)
        return self

    def transform(self, texts: pd.Series) -> sparse.csr_matrix:
        if not hasattr(self, "vectorizer_"):
            raise RuntimeError("TfidfFeatureExtractor must be fitted before transform")
        return self.vectorizer_.transform(texts)

    def parameter_grid(self) -> dict[str, list[Any]]:
        return {"feature_extractor__ngram_range": [(1, 1), (1, 2)]}

    def save_features(
        self,
        train_features: sparse.csr_matrix,
        test_features: sparse.csr_matrix,
        destination: Path,
    ) -> None:
        sparse.save_npz(
            destination / "tfidf_train.npz", train_features, compressed=True
        )
        sparse.save_npz(
            destination / "tfidf_test.npz", test_features, compressed=True
        )


@lru_cache(maxsize=None)
def _cached_glove_vectors(project_root: str, model_name: str) -> Any:
    cache_dir = Path(project_root) / ".cache" / "gensim"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("GENSIM_DATA_DIR", str(cache_dir))
    import gensim.downloader as api

    print(f"Loading pretrained embeddings: {model_name}")
    return api.load(model_name)


class GloveFeatureExtractor(FeatureExtractor):
    name = "glove"
    label = "GloVe-Twitter-100"
    file_stem = "glove_twitter_100"
    cache_features = True
    search_n_jobs = 1

    def __init__(
        self,
        project_root: Path | str = Path("."),
        model_name: str = GLOVE_MODEL_NAME,
    ) -> None:
        self.project_root = project_root
        self.model_name = model_name

    def _vectors(self) -> Any:
        project_root = str(Path(self.project_root).resolve())
        return _cached_glove_vectors(project_root, self.model_name)

    def fit(self, texts: pd.Series, y: Any = None) -> GloveFeatureExtractor:
        self._vectors()
        return self

    def transform(self, texts: pd.Series) -> np.ndarray:
        vectors = self._vectors()
        matrix = np.zeros(
            (len(texts), int(vectors.vector_size)), dtype=np.float32
        )
        for row_index, text in enumerate(texts):
            tokens = GLOVE_TOKEN_PATTERN.findall(str(text))
            known_tokens = [
                token for token in tokens if token in vectors.key_to_index
            ]
            if known_tokens:
                matrix[row_index] = np.mean(
                    vectors[known_tokens], axis=0, dtype=np.float32
                )
        return matrix

    def pipeline_steps(self) -> list[tuple[str, Any]]:
        return [
            ("feature_extractor", self),
            ("feature_scaler", StandardScaler()),
        ]

    def save_features(
        self,
        train_features: np.ndarray,
        test_features: np.ndarray,
        destination: Path,
    ) -> None:
        np.save(destination / "glove_twitter_100_train.npy", train_features)
        np.save(destination / "glove_twitter_100_test.npy", test_features)
