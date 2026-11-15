from __future__ import annotations

import html
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_SEED = 42
VALIDATION_SIZE = 0.20
TEXT_COLUMN = "OriginalTweet"
PROCESSED_TEXT_COLUMN = "ProcessedTweet"
TARGET_COLUMN = "Sentiment"

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
MENTION_PATTERN = re.compile(r"(?<!\w)@\w+")
HASHTAG_PATTERN = re.compile(r"(?<!\w)#\w+")


def normalise_for_deduplication(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def preprocess_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = html.unescape(str(value)).lower()
    text = text.replace("\x92", "'").replace("\u2019", "'")
    text = "".join(
        character
        for character in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(character)
    )
    text = URL_PATTERN.sub(" <url> ", text)
    text = MENTION_PATTERN.sub(" <user> ", text)
    text = HASHTAG_PATTERN.sub(lambda match: " " + match.group(0)[1:] + " ", text)
    return re.sub(r"\s+", " ", text).strip()


def prepare_data(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:

    train = train_raw.copy()
    test = test_raw.copy()
    train["_normalised_text"] = train[TEXT_COLUMN].map(normalise_for_deduplication)
    test["_normalised_text"] = test[TEXT_COLUMN].map(normalise_for_deduplication)

    test_texts = set(test["_normalised_text"])
    overlap_mask = train["_normalised_text"].isin(test_texts)
    overlapping_train_rows = int(overlap_mask.sum())
    train = train.loc[~overlap_mask].copy()

    duplicate_mask = train["_normalised_text"].duplicated(keep="first")
    duplicate_train_rows_removed = int(duplicate_mask.sum())
    train = train.loc[~duplicate_mask].copy()

    train[PROCESSED_TEXT_COLUMN] = train[TEXT_COLUMN].map(preprocess_text)
    test[PROCESSED_TEXT_COLUMN] = test[TEXT_COLUMN].map(preprocess_text)

    clean_training_rows = len(train)
    train, validation = train_test_split(
        train,
        test_size=VALIDATION_SIZE,
        stratify=train[TARGET_COLUMN],
        random_state=RANDOM_SEED,
    )
    audit = {
        "raw_training_rows": len(train_raw),
        "raw_test_rows": len(test_raw),
        "overlapping_training_rows_removed": overlapping_train_rows,
        "duplicate_training_rows_removed": duplicate_train_rows_removed,
        "clean_training_rows": clean_training_rows,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
    }
    return (
        train.reset_index(names="source_index"),
        validation.reset_index(names="source_index"),
        test.reset_index(names="source_index"),
        audit,
    )


def save_preparation_audit(audit: dict[str, int], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2)
