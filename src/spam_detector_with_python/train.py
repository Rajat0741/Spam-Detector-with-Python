"""Training pipeline."""

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from .data import load_dataset
from .model import evaluate_model, save_artifacts, train_model
from .preprocessing import preprocess_text


def train_and_evaluate(
    dataset_path: Path,
    artifacts_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
):
    df = load_dataset(dataset_path)
    df["message"] = df["message"].map(preprocess_text)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["message"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    save_artifacts(
        model,
        vectorizer,
        artifacts_dir / "model.pkl",
        artifacts_dir / "vectorizer.pkl",
    )
    return model, vectorizer, metrics