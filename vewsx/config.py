"""VewsX configuration — paths, constants, defaults."""

from __future__ import annotations

import os
from pathlib import Path

# Project root is the parent of vewsx/
PROJECT_ROOT = Path(__file__).parent.parent

# Default paths (can be overridden via environment variables)
DATA_DIR = Path(os.environ.get("VEWSX_DATA_DIR", str(PROJECT_ROOT / ".data")))
CACHE_DIR = Path(os.environ.get("VEWSX_CACHE_DIR", str(PROJECT_ROOT / ".cache")))
OUTPUTS_DIR = Path(os.environ.get("VEWSX_OUTPUTS_DIR", str(PROJECT_ROOT / "outputs")))

# MIND TSV column schemas
NEWS_COLUMNS = [
    "id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]

BEHAVIORS_COLUMNS = [
    "impression_id",
    "user_id",
    "time",
    "history",
    "impressions",
]

# Category color palette (deterministic mapping for consistency)
CATEGORY_COLORS = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
    "#1F77B4",
    "#FF7F0E",
    "#2CA02C",
    "#D62728",
    "#9467BD",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
    "#BCBD22",
    "#17BECF",
]
