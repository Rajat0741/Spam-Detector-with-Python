"""Command-line entry point."""

from pathlib import Path

from .model import load_artifacts
from .predict import predict_message
from .train import train_and_evaluate


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    dataset_path = project_root / "data" / "spam_dataset.csv"
    artifacts_dir = project_root / "artifacts"

    model, vectorizer, metrics = train_and_evaluate(dataset_path, artifacts_dir)

    print(f"Model accuracy: {metrics['accuracy']:.4f}")
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])

    # Reload from disk to verify persistence works end-to-end.
    model, vectorizer = load_artifacts(
        artifacts_dir / "model.pkl", artifacts_dir / "vectorizer.pkl"
    )
    print("Model and vectorizer have been loaded successfully.")

    test_message = input("Enter a message for classification: ")
    print("\nMessage:", test_message)
    print(f"The input Message is Classified as: {predict_message(model, vectorizer, test_message)}")


if __name__ == "__main__":
    main()