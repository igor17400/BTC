"""
Custom Dataset Utilities

This module provides functions to preprocess custom dataset formats
to match the expected MIND dataset format for NewsReX.

The MIND format expects:
- behaviors.tsv: impression_id, user_id, time, history, impressions (5 columns)
- news.tsv: id, category, subcategory, title, abstract, url, title_entities, abstract_entities (8 columns)
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Union
import ast

logger = logging.getLogger(__name__)


def convert_jp_behaviors_to_mind_format(
        input_file: Union[str, Path],
        output_file: Union[str, Path],
        user_id_prefix: str = "U",
        news_id_prefix: str = "N",
        impression_id_start: int = 1,
        time_format: str = "auto"
) -> None:
    """
    Convert a custom behaviors dataset to MIND format.
    
    Expected input format (12 columns):
    uid, impid, time, end_time, history, impressions, n_hist, n_impr, n_impr_1, n_impr_0, n_clicked, n_candidates
    
    Output format (5 columns):
    impression_id, user_id, time, history, impressions
    
    Args:
        input_file: Path to input behaviors file
        output_file: Path to output behaviors file
        user_id_prefix: Prefix for user IDs (default: "U")
        news_id_prefix: Prefix for news IDs (default: "N") 
        impression_id_start: Starting impression ID (default: 1)
        time_format: Time format conversion ("auto", "mind", or custom format string)
    """
    logger.info(f"Converting custom behaviors format from {input_file} to MIND format at {output_file}")

    # Read the input file - first try with headers, then without

    column_names = ['uid', 'impid', 'time', 'end_time', 'history', 'impressions',
                    'n_hist', 'n_impr', 'n_impr_1', 'n_impr_0', 'n_clicked', 'n_candidates']
    behaviors = pd.read_table(
        input_file,
        header=None,
        names=column_names,
        usecols=range(len(column_names)),
    )

    logger.info(f"Loaded {len(behaviors)} behaviors from japanese input file")
    logger.info(f"Input columns: {list(behaviors.columns)}")

    # Create output dataframe with MIND format
    output_df = pd.DataFrame()

    # 1. impression_id: Sequential ID starting from impression_id_start
    output_df['impression_id'] = range(impression_id_start, impression_id_start + len(behaviors))

    # 2. user_id: Convert uid to string with prefix
    output_df['user_id'] = user_id_prefix + behaviors['uid'].astype(str)

    # 3. time: Convert time format
    if time_format == "auto":
        # Try to detect and convert common time formats
        output_df['time'] = _convert_time_format(behaviors['time'])
    elif time_format == "mind":
        # MIND format: MM/dd/yyyy h:mm:ss AM/PM
        output_df['time'] = pd.to_datetime(behaviors['time']).dt.strftime('%m/%d/%Y %I:%M:%S %p')
    else:
        # Custom format string
        output_df['time'] = pd.to_datetime(behaviors['time']).dt.strftime(time_format)

    # 4. history: Convert list format to space-separated string
    output_df['history'] = behaviors['history'].apply(_convert_history_format)

    # 5. impressions: Convert list format to space-separated string
    output_df['impressions'] = behaviors['impressions'].apply(_convert_impressions_format)

    # Save to output file
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_df.to_csv(output_file, sep='\t', header=False, index=False)
    logger.info(f"Saved {len(output_df)} behaviors in MIND format to {output_file}")

    # Log sample of converted data
    logger.info("Sample of converted data:")
    logger.info(f"Input sample:\n{behaviors.head(2)}")
    logger.info(f"Output sample:\n{output_df.head(2)}")


def convert_custom_news_to_mind_format(
        input_file: Union[str, Path],
        output_file: Union[str, Path],
) -> None:
    """
    Convert a custom news dataset to MIND format.
    
    Expected input format (11 columns):
    nid, category, subcategory, title, full_text, abstract, URL, publish_date, publish_datetime, title_entities, abstract_entities
    
    Output format (8 columns):
    id, category, subcategory, title, abstract, url, title_entities, abstract_entities
    
    Args:
        input_file: Path to input news file
        output_file: Path to output news file
    """
    logger.info(f"Converting custom news format from {input_file} to MIND format at {output_file}")

    # Read the input file (skip header row)
    news = pd.read_table(input_file, header=0)  # Use header=0 to read with headers
    logger.info(f"Loaded {len(news)} news articles from input file")

    # The file already has the correct column names
    logger.info(f"Input columns: {list(news.columns)}")

    # Select and rename columns for MIND format (8 columns)
    # Map: nid -> id, abstract -> abstract, URL -> url
    output_news = news[['nid', 'category', 'subcategory', 'title', 'abstract', 'URL',
                        'title_entities', 'abstract_entities']].copy()

    # Rename columns to match MIND format
    output_news.columns = ['id', 'category', 'subcategory', 'title', 'abstract', 'url',
                           'title_entities', 'abstract_entities']

    # Save to output file
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_news.to_csv(output_file, sep='\t', header=False, index=False)
    logger.info(f"Saved {len(output_news)} news articles in MIND format to {output_file}")


def _convert_time_format(time_series: pd.Series) -> pd.Series:
    """Convert time series to MIND format (MM/dd/yyyy h:mm:ss AM/PM)."""
    try:
        # Try to parse as datetime and convert to MIND format
        dt_series = pd.to_datetime(time_series)
        return dt_series.dt.strftime('%m/%d/%Y %I:%M:%S %p')
    except Exception as e:
        logger.warning(f"Failed to convert time format: {e}. Returning as-is.")
        return time_series


def _convert_history_format(history_item) -> str:
    """Convert history from list format to space-separated string."""
    if pd.isna(history_item) or history_item == '[]' or history_item == []:
        return ""

    try:
        # If it's a string representation of a list, parse it
        if isinstance(history_item, str):
            if history_item.startswith('[') and history_item.endswith(']'):
                history_list = ast.literal_eval(history_item)
            else:
                # Already space-separated
                return history_item
        else:
            history_list = history_item

        # Convert list to space-separated string
        if isinstance(history_list, list):
            return " ".join(str(item) for item in history_list)
        else:
            return str(history_list)

    except Exception as e:
        logger.warning(f"Failed to convert history format for {history_item}: {e}")
        return str(history_item) if not pd.isna(history_item) else ""


def _convert_impressions_format(impressions_item) -> str:
    """Convert impressions from list format to space-separated string."""
    if pd.isna(impressions_item):
        return ""

    try:
        # If it's a string representation of a list, parse it
        if isinstance(impressions_item, str):
            if impressions_item.startswith('[') and impressions_item.endswith(']'):
                impressions_list = ast.literal_eval(impressions_item)
            else:
                # Already space-separated
                return impressions_item
        else:
            impressions_list = impressions_item

        # Convert list to space-separated string
        if isinstance(impressions_list, list):
            return " ".join(str(item) for item in impressions_list)
        else:
            return str(impressions_list)

    except Exception as e:
        logger.warning(f"Failed to convert impressions format for {impressions_item}: {e}")
        return str(impressions_item) if not pd.isna(impressions_item) else ""



def preprocess_custom_dataset(
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        behaviors_filename: str = "behaviors.tsv",
        news_filename: str = "news.tsv",
        user_id_prefix: str = "U",
        news_id_prefix: str = "N",
        time_format: str = "mind"
) -> None:
    """
    Preprocess an entire custom dataset directory to MIND format.
    
    Args:
        input_dir: Directory containing custom format files
        output_dir: Directory to save MIND format files
        behaviors_filename: Name of behaviors file (default: "behaviors.tsv")
        news_filename: Name of news file (default: "news.tsv")
        user_id_prefix: Prefix for user IDs (default: "U")
        news_id_prefix: Prefix for news IDs (default: "N")
        time_format: Time format for conversion (default: "mind")
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    logger.info(f"Preprocessing custom dataset from {input_path} to {output_path}")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Convert behaviors file
    behaviors_input = input_path / behaviors_filename
    behaviors_output = output_path / behaviors_filename

    if behaviors_input.exists():
        convert_jp_behaviors_to_mind_format(
            behaviors_input,
            behaviors_output,
            user_id_prefix=user_id_prefix,
            news_id_prefix=news_id_prefix,
            time_format=time_format
        )
    else:
        logger.warning(f"Behaviors file not found: {behaviors_input}")

    # Convert news file
    news_input = input_path / news_filename
    news_output = output_path / news_filename

    if news_input.exists():
        convert_custom_news_to_mind_format(
            news_input,
            news_output
        )
    else:
        logger.warning(f"News file not found: {news_input}")

    logger.info(f"Dataset preprocessing completed. Files saved to {output_path}")
