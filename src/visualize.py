import logging
from pathlib import Path

import hydra
import plotly.io as pio
from omegaconf import DictConfig
from utils.analytics import NewsRecAnalytics

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config")
def visualize(cfg: DictConfig) -> None:
    """Generate visualizations and analytics"""
    # Create output directory
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)

    # Load data
    dataset = hydra.utils.instantiate(cfg.dataset)
    news_df, behaviors_df = dataset.load_raw_data()

    # Initialize analytics
    analytics = NewsRecAnalytics(news_df, behaviors_df)

    # Generate visualizations
    logger.info("Generating long-tail distribution plot...")
    long_tail_fig = analytics.plot_long_tail_distribution()
    pio.write_html(long_tail_fig, output_dir / "long_tail_distribution.html")

    logger.info("Generating category insights...")
    category_fig = analytics.generate_category_insights()
    pio.write_html(category_fig, output_dir / "category_insights.html")

    # Generate user profiles
    sample_users = behaviors_df["user_id"].sample(n=10).unique()
    for user_id in sample_users:
        logger.info(f"Generating profile for user {user_id}...")
        user_fig = analytics.create_user_profile_visualization(user_id)
        pio.write_html(user_fig, output_dir / f"user_profile_{user_id}.html")

    # Generate recommendation diversity analysis
    logger.info("Analyzing recommendation diversity...")
    diversity_fig = analytics.plot_recommendation_diversity(
        dataset.get_recommendations()
    )
    pio.write_html(diversity_fig, output_dir / "recommendation_diversity.html")

    # Generate user journeys
    for user_id in sample_users:
        logger.info(f"Generating journey for user {user_id}...")
        journey_fig = analytics.create_interactive_user_journey(user_id)
        pio.write_html(journey_fig, output_dir / f"user_journey_{user_id}.html")


if __name__ == "__main__":
    visualize()
