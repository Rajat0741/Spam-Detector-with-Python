"""Backwards-compatible shim for the original module name."""

from .cli import main
from .preprocessing import preprocess_text

__all__ = ["main", "preprocess_text"]

if __name__ == "__main__":
    main()