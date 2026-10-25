"""Prediction helpers."""

from .preprocessing import preprocess_text


def predict_message(model, vectorizer, message: str) -> str:
    preprocessed = preprocess_text(message)
    vector = vectorizer.transform([preprocessed])
    prediction = model.predict(vector)[0]
    return "Spam" if prediction == 1 else "Not Spam"