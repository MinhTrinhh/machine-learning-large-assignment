from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


SENTIMENT_COLORS = ["#8b1e3f", "#d95f59", "#a7a7a7", "#57a773", "#16697a"]
FEATURE_SPACE_SAMPLE_SIZE = 5_000
FEATURE_SPACE_RANDOM_SEED = 42


def _save_current_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    if "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close()


def _stratified_sample_indices(labels: np.ndarray, sample_size: int) -> np.ndarray:
    if sample_size >= len(labels):
        return np.arange(len(labels))
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=sample_size,
        random_state=FEATURE_SPACE_RANDOM_SEED,
    )
    indices, _ = next(splitter.split(np.zeros(len(labels)), labels))
    return np.sort(indices)


def _plot_feature_embedding(
    coordinates: np.ndarray,
    labels: np.ndarray,
    title: str,
    axis_labels: tuple[str, str],
    destination: Path,
) -> None:
    sentiment_order = [
        "Extremely Negative",
        "Negative",
        "Neutral",
        "Positive",
        "Extremely Positive",
    ]
    palette = dict(zip(sentiment_order, SENTIMENT_COLORS))
    plot_data = pd.DataFrame(
        {"x": coordinates[:, 0], "y": coordinates[:, 1], "sentiment": labels}
    )
    sns.set_theme(style="whitegrid", context="notebook")
    plt.figure(figsize=(9.5, 7.2))
    axis = sns.scatterplot(
        data=plot_data,
        x="x",
        y="y",
        hue="sentiment",
        hue_order=sentiment_order,
        palette=palette,
        s=15,
        alpha=0.58,
        linewidth=0,
    )
    axis.set(title=title, xlabel=axis_labels[0], ylabel=axis_labels[1])
    axis.legend(
        title="Sentiment",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        markerscale=1.6,
    )
    _save_current_figure(destination)


def plot_feature_space_visualizations(
    project_root: Path | str = Path("."),
    representations: dict[str, tuple[Any, str]] | None = None,
    labels: np.ndarray | pd.Series | None = None,
    sample_size: int = FEATURE_SPACE_SAMPLE_SIZE,
) -> dict[str, Any]:

    try:
        import umap
    except ImportError as error:
        raise ImportError(
        ) from error

    root = Path(project_root).resolve()
    feature_dir = root / "features"
    figure_dir = root / "outputs" / "graphics" / "feature_space"
    if labels is None:
        label_array = np.load(feature_dir / "train_labels.npy", allow_pickle=True)
    else:
        label_array = np.asarray(labels)
    sample_indices = _stratified_sample_indices(label_array, sample_size)
    sampled_labels = label_array[sample_indices]

    if representations is None:
        representations = {
            "tfidf": (sparse.load_npz(feature_dir / "tfidf_train.npz"), "TF-IDF"),
            "glove": (
                np.load(feature_dir / "glove_twitter_100_train.npy"),
                "GloVe-Twitter-100",
            ),
        }
    if not representations:
        raise ValueError("Provide at least one feature representation to visualise")

    sampled_representations: dict[str, tuple[Any, str]] = {}
    for feature_name, (matrix, feature_label) in representations.items():
        if matrix.shape[0] != len(label_array):
            raise ValueError(
                f"{feature_label} has {matrix.shape[0]} rows but labels have "
                f"{len(label_array)} rows"
            )
        sampled_matrix = matrix[sample_indices]
        if feature_name == "glove":
            sampled_matrix = StandardScaler().fit_transform(sampled_matrix)
        sampled_representations[feature_name] = (sampled_matrix, feature_label)

    metadata: dict[str, Any] = {
        "source_split": "clean training set only",
        "sample_size": int(len(sample_indices)),
        "sampling": "class-stratified without replacement",
        "random_seed": FEATURE_SPACE_RANDOM_SEED,
        "shared_sample_across_representations": True,
        "tsne": {"perplexity": 30, "max_iter": 1_000, "input_dimensions": 50},
        "umap": {
            "n_neighbors": 15,
            "min_dist": 0.1,
            "metric": "cosine",
            "input_dimensions": 50,
        },
        "representations": {},
    }

    for feature_name, (matrix, feature_label) in sampled_representations.items():
        reducer = (
            TruncatedSVD(n_components=50, random_state=FEATURE_SPACE_RANDOM_SEED)
            if sparse.issparse(matrix)
            else PCA(n_components=50, random_state=FEATURE_SPACE_RANDOM_SEED)
        )
        reduced = reducer.fit_transform(matrix)
        explained = np.asarray(reducer.explained_variance_ratio_)
        linear_method = "Truncated SVD (sparse PCA analogue)" if feature_name == "tfidf" else "PCA"
        output_dir = figure_dir / feature_name

        _plot_feature_embedding(
            reduced[:, :2],
            sampled_labels,
            f"{feature_label} feature space: {linear_method}",
            (
                f"Component 1 ({explained[0] * 100:.1f}% variance)",
                f"Component 2 ({explained[1] * 100:.1f}% variance)",
            ),
            output_dir / "pca.png",
        )

        tsne_coordinates = TSNE(
            n_components=2,
            perplexity=30,
            init="pca",
            learning_rate="auto",
            max_iter=1_000,
            random_state=FEATURE_SPACE_RANDOM_SEED,
        ).fit_transform(reduced)
        _plot_feature_embedding(
            tsne_coordinates,
            sampled_labels,
            f"{feature_label} feature space: t-SNE",
            ("t-SNE dimension 1", "t-SNE dimension 2"),
            output_dir / "tsne.png",
        )

        umap_coordinates = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            random_state=FEATURE_SPACE_RANDOM_SEED,
            n_jobs=1,
            compatibility_layout=True,
        ).fit_transform(reduced)
        _plot_feature_embedding(
            umap_coordinates,
            sampled_labels,
            f"{feature_label} feature space: UMAP",
            ("UMAP dimension 1", "UMAP dimension 2"),
            output_dir / "umap.png",
        )

        metadata["representations"][feature_name] = {
            "feature_label": feature_label,
            "linear_method": linear_method,
            "component_1_explained_variance": float(explained[0]),
            "component_2_explained_variance": float(explained[1]),
            "first_50_components_explained_variance": float(explained.sum()),
        }

    metadata_path = root / "outputs" / "generated" / "feature_space_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    return metadata


def plot_eda_results(
    train: pd.DataFrame,
    class_counts: pd.DataFrame,
    unigrams: pd.DataFrame,
    bigrams: pd.DataFrame,
    sentiment_order: list[str],
    figure_dir: Path,
) -> None:
    """Create and save every report-ready EDA figure."""
    sns.set_theme(style="whitegrid", context="notebook")
    palette = dict(zip(sentiment_order, SENTIMENT_COLORS))

    plt.figure(figsize=(10, 5.5))
    axis = sns.barplot(
        data=class_counts,
        x="sentiment",
        y="count",
        hue="sentiment",
        order=sentiment_order,
        palette=palette,
        legend=False,
    )
    axis.set(title="Training-set sentiment distribution", xlabel="Sentiment", ylabel="Tweets")
    axis.tick_params(axis="x", rotation=20)
    for container in axis.containers:
        axis.bar_label(container, fmt="%d", padding=3)
    _save_current_figure(figure_dir / "class_distribution.png")

    _, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(train, x="word_count", bins=50, ax=axes[0], color="#16697a")
    axes[0].set(title="Tweet length in words", xlabel="Words per tweet", ylabel="Tweets")
    sns.histplot(train, x="char_count", bins=50, ax=axes[1], color="#d95f59")
    axes[1].set(title="Tweet length in characters", xlabel="Characters per tweet", ylabel="Tweets")
    _save_current_figure(figure_dir / "text_length_distributions.png")

    plt.figure(figsize=(11, 5.5))
    axis = sns.boxplot(
        data=train,
        x="Sentiment",
        y="word_count",
        order=sentiment_order,
        hue="Sentiment",
        palette=palette,
        legend=False,
        showfliers=False,
    )
    axis.set(title="Tweet word count by sentiment", xlabel="Sentiment", ylabel="Words per tweet")
    axis.tick_params(axis="x", rotation=20)
    _save_current_figure(figure_dir / "word_count_by_sentiment.png")

    for table, title, filename, color in (
        (unigrams, "Most frequent training-set words", "top_unigrams.png", "#16697a"),
        (bigrams, "Most frequent training-set bigrams", "top_bigrams.png", "#57a773"),
    ):
        plt.figure(figsize=(10, 7))
        axis = sns.barplot(data=table, x="count", y="term", color=color)
        axis.set(title=title, xlabel="Frequency", ylabel="")
        _save_current_figure(figure_dir / filename)


def plot_confusion_matrix(
    matrix: np.ndarray,
    label_order: list[str],
    model_name: str,
    split_name: str,
    destination: Path,
) -> None:
    """Save a labelled confusion-matrix heatmap."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_order,
        yticklabels=label_order,
        cbar=False,
    )
    plt.title(
        f"{model_name.replace('_', ' ').title()} - {split_name} confusion matrix"
    )
    plt.xlabel("Predicted sentiment")
    plt.ylabel("True sentiment")
    plt.xticks(rotation=25, ha="right")
    plt.yticks(rotation=0)
    _save_current_figure(destination)


def plot_model_comparison(metrics: pd.DataFrame, destination: Path) -> None:
    """Compare accuracy and macro-F1 across representations and classifiers."""
    sns.set_theme(style="whitegrid", context="notebook")
    plot_data = metrics.melt(
        id_vars=["feature_method", "model"],
        value_vars=["accuracy", "macro_f1"],
        var_name="metric",
        value_name="score",
    )
    plot_data["model"] = plot_data["model"].str.replace("_", " ").str.title()
    grid = sns.catplot(
        data=plot_data,
        x="model",
        y="score",
        hue="feature_method",
        col="metric",
        kind="bar",
        height=5.2,
        aspect=0.95,
        sharey=True,
    )
    grid.set_axis_labels("Classifier", "Score")
    grid.set_titles("{col_name}")
    grid.set(ylim=(0, 1))
    grid.figure.suptitle(
        "TF-IDF and GloVe performance on the official test set", y=1.03
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    grid.savefig(destination, dpi=180, bbox_inches="tight")
    if "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close(grid.figure)


def plot_experiment_results(
    result: dict[str, Any],
    y_true: pd.Series,
    project_root: Path | str = Path("."),
) -> None:
    """Plot every model's confusion matrix and the cross-feature comparison."""
    root = Path(project_root).resolve()
    feature_name = str(result["feature_name"])
    feature_label = str(result["feature_label"])
    label_order = list(result["label_order"])
    for model_name, predictions in result["test_predictions"].items():
        matrix = confusion_matrix(y_true, predictions, labels=label_order)
        plot_confusion_matrix(
            matrix,
            label_order,
            model_name,
            f"{feature_label} test",
            root
            / "outputs"
            / "graphics"
            / feature_name
            / f"{model_name}_test_confusion_matrix.png",
        )

