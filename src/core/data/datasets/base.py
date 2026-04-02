"""Abstract base class for news recommendation datasets."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class BaseNewsDataset(ABC):
    """Abstract base for news recommendation datasets.

    Subclasses must implement data loading and processing.
    Dataloader creation is framework-specific and handled by runners.
    """

    def __init__(self) -> None:
        self.news_data: dict[str, np.ndarray] = {}
        self.train_behaviors_data: dict[str, Any] = {}
        self.val_behaviors_data: dict[str, Any] = {}
        self.test_behaviors_data: dict[str, Any] = {}
        self.root_dir = Path(".")

    @abstractmethod
    def download_dataset(self) -> None:
        """Download and extract dataset files."""

    @abstractmethod
    def process_news(self) -> dict[str, Any]:
        """Process news articles into numerical format."""

    @abstractmethod
    def get_train_val_data(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Get processed training and validation data."""

    @abstractmethod
    def get_test_data(self) -> dict[str, Any]:
        """Get processed test data."""

    def download_and_extract(self, url: str, split: str) -> None:
        """Helper to download and extract dataset zip files."""
        import urllib.request
        import zipfile

        zip_path = self.root_dir / f"{split}.zip"
        extract_path = self.root_dir / split

        if not extract_path.exists():
            print(f"Downloading {split} set...")
            urllib.request.urlretrieve(url, zip_path)

            print(f"Extracting {split} set...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_path)

            zip_path.unlink()
