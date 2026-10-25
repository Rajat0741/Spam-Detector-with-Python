"""Text preprocessing helpers for the spam detector."""

import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

_stop_words: set[str] | None = None
_lemmatizer: WordNetLemmatizer | None = None


def ensure_nltk_resources() -> None:
    """Download the NLTK corpora required for preprocessing."""
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)


def _get_stop_words() -> set[str]:
    global _stop_words
    if _stop_words is None:
        ensure_nltk_resources()
        _stop_words = set(stopwords.words("english"))
    return _stop_words


def _get_lemmatizer() -> WordNetLemmatizer:
    global _lemmatizer
    if _lemmatizer is None:
        ensure_nltk_resources()
        _lemmatizer = WordNetLemmatizer()
    return _lemmatizer


def preprocess_text(text: str) -> str:
    """Lowercase, strip punctuation, remove stopwords, and lemmatize."""
    text = text.lower()
    text = re.sub(r"\W", " ", text)
    stop_words = _get_stop_words()
    lemmatizer = _get_lemmatizer()
    words = [word for word in text.split() if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)