from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

if __package__:
    from modules.data_downloader import get_dataset_metadata, load_dataset
    from modules.data_preprocessor import (
        HASHTAG_PATTERN,
        MENTION_PATTERN,
        URL_PATTERN,
    )
    from modules.plotter import plot_eda_results
else:  # Support direct execution: python modules/text_eda.py
    from data_downloader import get_dataset_metadata, load_dataset
    from data_preprocessor import HASHTAG_PATTERN, MENTION_PATTERN, URL_PATTERN
    from plotter import plot_eda_results


SENTIMENT_ORDER = [
    "Extremely Negative",
    "Negative",
    "Neutral",
    "Positive",
    "Extremely Positive",
]
# Compact stop-word list for descriptive frequency plots only. Negators are
# deliberately retained because they carry sentiment information.
EDA_STOPWORDS = {
    "a", "about", "after", "again", "all", "am", "an", "and", "any",
    "are", "as", "at", "be", "because", "been", "before", "being", "but",
    "by", "can", "could", "did", "do", "does", "doing", "down", "during",
    "each", "few", "for", "from", "further", "had", "has", "have", "having",
    "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
    "more", "most", "my", "myself", "of", "off", "on", "once", "only",
    "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same",
    "she", "should", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "will", "with", "would", "you", "your", "yours", "yourself",
    "yourselves", "amp", "rt",
}
TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?")


def _normalise_for_comparison(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _tokens(value: object) -> list[str]:
    text = html.unescape(str(value)).lower()
    # The source uses legacy punctuation bytes in some contractions.
    text = text.replace("\x92", "'").replace("\u2019", "'")
    text = URL_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)
    return [
        token
        for token in TOKEN_PATTERN.findall(text)
        if len(token) > 1 and token not in EDA_STOPWORDS
    ]


def add_descriptive_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    text = result["OriginalTweet"].fillna("").astype(str)
    result["char_count"] = text.str.len()
    result["word_count"] = text.str.findall(r"\b\w+\b").str.len()
    result["url_count"] = text.str.count(URL_PATTERN)
    result["mention_count"] = text.str.count(MENTION_PATTERN)
    result["hashtag_count"] = text.str.count(HASHTAG_PATTERN)
    return result


def build_quality_table(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, frame in (("train", train), ("test", test)):
        text = frame["OriginalTweet"]
        normalised = text.fillna("").map(_normalise_for_comparison)
        duplicate_mask = normalised.duplicated(keep=False) & normalised.ne("")
        conflicting = (
            frame.loc[duplicate_mask]
            .assign(_normalised=normalised.loc[duplicate_mask])
            .groupby("_normalised")["Sentiment"]
            .nunique()
            .gt(1)
            .sum()
        )
        rows.append(
            {
                "split": split,
                "rows": len(frame),
                "columns": len(frame.columns),
                "missing_text": int(text.isna().sum()),
                "missing_sentiment": int(frame["Sentiment"].isna().sum()),
                "empty_or_whitespace_text": int(normalised.eq("").sum()),
                "exact_duplicate_rows": int(frame.duplicated().sum()),
                "duplicated_text_rows": int(duplicate_mask.sum()),
                "duplicate_text_groups_with_conflicting_labels": int(conflicting),
            }
        )
    train_text = set(train["OriginalTweet"].fillna("").map(_normalise_for_comparison))
    test_text = set(test["OriginalTweet"].fillna("").map(_normalise_for_comparison))
    overlap = len((train_text & test_text) - {""})
    for row in rows:
        row["unique_train_test_text_overlap"] = overlap
    return pd.DataFrame(rows)


def _frequency_table(token_lists: pd.Series, ngram_size: int, top_n: int = 25) -> pd.DataFrame:
    counts: Counter[object] = Counter()
    for tokens in token_lists:
        if ngram_size == 1:
            counts.update(tokens)
        else:
            counts.update(zip(tokens, tokens[1:]))
    rows = []
    for term, count in counts.most_common(top_n):
        label = term if isinstance(term, str) else " ".join(term)
        rows.append({"term": label, "count": count})
    return pd.DataFrame(rows)


def run_eda(
    project_root: Path | str = Path("."),
    show_sample_rows: bool = True,
) -> dict[str, object]:
    """Run the complete EDA and save tables and figures under outputs/."""
    root = Path(project_root).resolve()
    data_dir = root / "data"
    figure_dir = root / "outputs" / "graphics" / "eda"
    table_dir = root / "outputs" / "generated" / "eda"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    train_raw, test_raw = load_dataset(data_dir)
    train = add_descriptive_features(train_raw)

    print(f"Training shape: {train_raw.shape}")
    print(f"Test shape:     {test_raw.shape}")
    print(f"Columns: {list(train_raw.columns)}")
    if show_sample_rows:
        display_columns = ["TweetAt", "OriginalTweet", "Sentiment"]
        print("\nTraining sample:")
        print(train_raw[display_columns].head(3).to_string(index=False))

    quality = build_quality_table(train_raw, test_raw)
    quality.to_csv(table_dir / "data_quality_summary.csv", index=False)
    print("\nData quality summary:")
    print(quality.to_string(index=False))

    class_counts = (
        train["Sentiment"]
        .value_counts()
        .reindex(SENTIMENT_ORDER, fill_value=0)
        .rename_axis("sentiment")
        .reset_index(name="count")
    )
    class_counts["percentage"] = class_counts["count"] / len(train) * 100
    class_counts.to_csv(table_dir / "class_distribution.csv", index=False)

    token_lists = train["OriginalTweet"].fillna("").map(_tokens)
    unigrams = _frequency_table(token_lists, ngram_size=1)
    bigrams = _frequency_table(token_lists, ngram_size=2)

    plot_eda_results(
        train=train,
        class_counts=class_counts,
        unigrams=unigrams,
        bigrams=bigrams,
        sentiment_order=SENTIMENT_ORDER,
        figure_dir=figure_dir,
    )

    summary = {
        **get_dataset_metadata(),
        "train_rows": len(train_raw),
        "test_rows": len(test_raw),
        "sentiments": SENTIMENT_ORDER,
        "detailed_eda_split": "train only",
        "test_usage": "schema and integrity checks only",
        "figure_directory": str(figure_dir.relative_to(root)),
        "table_directory": str(table_dir.relative_to(root)),
        "frequency_plot_note": (
            "URLs, mentions, HTML entities, and a compact stop-word list were removed "
            "for descriptive frequency plots only; this is not the modeling preprocessing pipeline."
        ),
    }
    with (table_dir / "eda_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=True)

    print(f"\nSaved figures to: {figure_dir}")
    print(f"Saved two summary tables to: {table_dir}")
    return {
        "train": train,
        "test": test_raw,
        "quality": quality,
        "class_distribution": class_counts,
        "top_unigrams": unigrams,
        "top_bigrams": bigrams,
        "metadata": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (defaults to the parent of modules/).",
    )
    args = parser.parse_args()
    run_eda(args.project_root)


if __name__ == "__main__":
    main()
