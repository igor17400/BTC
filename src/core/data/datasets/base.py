import urllib.request
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


class BaseNewsDataset(ABC):
    """Abstract base class for news recommendation datasets"""

    def __init__(self) -> None:
        self.news_data: Dict[str, np.ndarray] = {}  # Store news features
        self.train_behaviors_data: Dict[
            str, np.ndarray
        ] = {}  # Store training behaviors data
        self.val_behaviors_data: Dict[
            str, np.ndarray
        ] = {}  # Store validation behaviors data
        self.test_behaviors_data: Dict[
            str, np.ndarray
        ] = {}  # Store test behaviors data
        self.root_dir = Path(".")  # Define root directory

    @abstractmethod
    def download_dataset(self) -> None:
        """Download and extract dataset files"""
        pass

    @abstractmethod
    def process_news(self, news_df: pd.DataFrame, split: str) -> Dict[str, np.ndarray]:
        """Process news articles into numerical format"""
        pass

    @abstractmethod
    def process_behaviors(
        self, behaviors_df: pd.DataFrame, news_dict: Dict[str, int], stage: str
    ) -> Dict[str, np.ndarray]:
        """Process user behaviors into numerical format"""
        pass

    @abstractmethod
    def get_train_val_data(self) -> Tuple[Tuple, Tuple]:
        """Get processed training and validation data

        Returns:
            Tuple containing:
            - (news_data, train_behaviors_data)
            - (news_data, val_behaviors_data)
        """
        pass

    @abstractmethod
    def get_test_data(self) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Get processed test data

        Returns:
            Tuple containing:
            - test_news_data: Dictionary of news features
            - test_behaviors_data: Dictionary of behaviors data
        """
        pass

    @abstractmethod
    def train_dataloader(self, batch_size: int):
        """Create training data loader"""
        pass

    @abstractmethod
    def user_history_dataloader(self, mode: str):
        """Create user history data loader"""
        pass

    @abstractmethod
    def news_dataloader(self):
        """Create processed news data loader"""
        pass

    def download_and_extract(self, url: str, split: str) -> None:
        """Helper method to download and extract dataset files"""
        zip_path = self.root_dir / f"{split}.zip"
        extract_path = self.root_dir / split

        if not extract_path.exists():
            print(f"Downloading {split} set...")
            urllib.request.urlretrieve(url, zip_path)

            print(f"Extracting {split} set...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_path)

            zip_path.unlink()
