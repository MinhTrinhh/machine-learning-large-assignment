from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd


DATASET_REVISION = "5e441292c9120525a4ddd54eba3d65187d66bf49"
DATASET_NAME = "Coronavirus tweets NLP - Text Classification"
ORIGINAL_SOURCE = (
    "https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification"
)
BASE_URL = (
    "https://huggingface.co/datasets/ImpactTom6819/covid_tweets_practice/"
    f"resolve/{DATASET_REVISION}"
)
FILES = {
    "train": {
        "name": "Corona_NLP_train.csv",
        "sha256": "684ff5f7382c7de2a3beb06fdc8d15520ab982f08b8124254b0c62586d146915",
        "rows": 41_157,
    },
    "test": {
        "name": "Corona_NLP_test.csv",
        "sha256": "73ae36853128f9013c1c44b219ffe61aa7c6b8bc643b1566f97564a454a42dd2",
        "rows": 3_798,
    },
}
EXPECTED_COLUMNS = {
    "UserName",
    "ScreenName",
    "Location",
    "TweetAt",
    "OriginalTweet",
    "Sentiment",
}


def get_dataset_metadata() -> dict[str, str]:
    return {
        "dataset": DATASET_NAME,
        "original_source": ORIGINAL_SOURCE,
        "download_revision": DATASET_REVISION,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(data_dir: Path) -> dict[str, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    for split, metadata in FILES.items():
        destination = data_dir / str(metadata["name"])
        expected_hash = str(metadata["sha256"])

        if not destination.exists() or _sha256(destination) != expected_hash:
            temporary = destination.with_suffix(destination.suffix + ".part")
            if temporary.exists():
                temporary.unlink()

            print(f"Downloading {metadata['name']} ...")
            urlretrieve(f"{BASE_URL}/{metadata['name']}", temporary)

            actual_hash = _sha256(temporary)
            if actual_hash != expected_hash:
                temporary.unlink(missing_ok=True)
                raise ValueError(
                    f"Checksum mismatch for {metadata['name']}: {actual_hash}"
                )
            temporary.replace(destination)

        paths[split] = destination

    return paths


def load_dataset(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = download_dataset(data_dir)
    frames = {
        split: pd.read_csv(path, encoding="ISO-8859-1")
        for split, path in paths.items()
    }

    for split, frame in frames.items():
        missing_columns = EXPECTED_COLUMNS.difference(frame.columns)
        if missing_columns:
            raise ValueError(f"{split} is missing columns: {sorted(missing_columns)}")

        expected_rows = int(FILES[split]["rows"])
        if len(frame) != expected_rows:
            raise ValueError(
                f"{split} has {len(frame):,} rows; expected {expected_rows:,}"
            )

    return frames["train"], frames["test"]
