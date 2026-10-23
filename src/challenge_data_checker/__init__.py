"""challenge-data-checker: audit ML benchmarking train/test splits for chemical data leakage."""

from importlib.metadata import version

from challenge_data_checker.api import audit

__version__ = version("challenge-data-checker")

__all__ = ["__version__", "audit"]
