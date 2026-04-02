import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from rich.console import Console

logger = logging.getLogger("cache_manager")
console = Console()


class CacheManager:
    """Manages dataset and model caches"""

    def __init__(self, cache_dir: str = ".cache"):
        # Get project root directory (src/utils/data/cache_manager.py -> project root)
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.cache_dir = self.project_root / cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.dataset_dir = self.project_root / "data"  # Datasets in project root/data
        self.embeddings_dir = self.cache_dir / "embeddings"  # Embeddings in .cache

        self.dataset_dir.mkdir(exist_ok=True)
        self.embeddings_dir.mkdir(exist_ok=True)

        # Load cache index
        self.index_file = self.cache_dir / "cache_index.json"
        self.cache_index = self._load_cache_index()

    def _load_cache_index(self) -> Dict:
        """Load or create cache index"""
        if self.index_file.exists():
            with open(self.index_file, "r") as f:
                return json.load(f)
        return {"datasets": {}, "embeddings": {}}

    def _save_cache_index(self) -> None:
        """Save cache index"""
        with open(self.index_file, "w") as f:
            json.dump(self.cache_index, f, indent=2)

    def get_dataset_path(self, dataset_name: str, version: str) -> Path:
        """Get path for dataset cache"""
        return self.dataset_dir / dataset_name / version

    def get_embedding_path(self, embedding_name: str, dim: int) -> Path:
        """Get path for embedding cache"""
        return self.embeddings_dir / f"{embedding_name}_{dim}d"

    def is_dataset_cached(self, dataset_name: str, version: str) -> bool:
        """Check if dataset is cached"""
        cache_path = self.get_dataset_path(dataset_name, version)
        return cache_path.exists() and f"{dataset_name}/{version}" in self.cache_index["datasets"]

    def is_embedding_cached(self, embedding_name: str, dim: int) -> bool:
        """Check if embedding is cached"""
        cache_path = self.get_embedding_path(embedding_name, dim)
        return cache_path.exists() and f"{embedding_name}_{dim}d" in self.cache_index["embeddings"]

    def add_to_cache(
        self, name: str, version: str, cache_type: str, metadata: Optional[Dict] = None
    ) -> None:
        """Add entry to cache index"""
        if cache_type == "dataset":
            self.cache_index["datasets"][f"{name}/{version}"] = {
                "added": datetime.now().isoformat(),
                "metadata": metadata or {},
            }
        elif cache_type == "embedding":
            self.cache_index["embeddings"][f"{name}_{version}d"] = {
                "added": datetime.now().isoformat(),
                "metadata": metadata or {},
            }
        self._save_cache_index()

    def clear_cache(self, cache_type: Optional[str] = None) -> None:
        """Clear cache files"""
        if cache_type == "datasets" or cache_type is None:
            logger.info("Clearing dataset cache...")
            # Don't delete the dataset files, just clear the index
            self.cache_index["datasets"] = {}

        if cache_type == "embeddings" or cache_type is None:
            logger.info("Clearing embeddings cache...")
            for path in self.embeddings_dir.glob("**/*"):
                if path.is_file():
                    path.unlink()
            self.cache_index["embeddings"] = {}

        self._save_cache_index()
