"""Model training, evaluation, and persistence helpers."""

import pickle
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB


def train_model(X, y) -> MultinomialNB:
    model = MultinomialNB()
    model.fit(X, y)
    return model


def evaluate_model(model, X, y) -> dict:
    predictions = model.predict(X)
    return {
        "accuracy": accuracy_score(y, predictions),
        "confusion_matrix": confusion_matrix(y, predictions),
        "report": classification_report(y, predictions),
    }


def save_artifacts(model, vectorizer, model_path: Path, vectorizer_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as model_file, open(vectorizer_path, "wb") as vectorizer_file:
        pickle.dump(model, model_file)
        pickle.dump(vectorizer, vectorizer_file)


def load_artifacts(model_path: Path, vectorizer_path: Path):
    with open(model_path, "rb") as model_file, open(vectorizer_path, "rb") as vectorizer_file:
        return pickle.load(model_file), pickle.load(vectorizer_file)