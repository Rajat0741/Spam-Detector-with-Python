"""Dataset loading helpers."""

from pathlib import Path

import pandas as pd


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the spam dataset and normalise its columns."""
    df = pd.read_csv(path, encoding="latin-1")
    df = df.rename(columns={"Message": "message", "Category": "label"})

    required = {"message", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns {sorted(required)}")

    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    if df["label"].isna().any():
        raise ValueError("Dataset contains unknown labels; expected 'ham' or 'spam'")
    df["label"] = df["label"].astype(int)
    return df