"""VewsX — Entry point for the Streamlit visualization platform.

Launch with: uv run streamlit run vewsx/app.py
"""

import streamlit as st

st.set_page_config(
    page_title="VewsX",
    page_icon=":material/newsmode:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Raw dataset pages --
overview = st.Page(
    "pages/raw/overview.py", title="Overview", icon=":material/dashboard:", default=True
)
news_corpus = st.Page(
    "pages/raw/news_corpus.py", title="News Corpus", icon=":material/article:"
)
temporal = st.Page(
    "pages/raw/temporal.py", title="Temporal Analysis", icon=":material/schedule:"
)
user_behavior = st.Page(
    "pages/raw/user_behavior.py", title="User Behavior", icon=":material/person:"
)
user_segments = st.Page(
    "pages/raw/user_segments.py", title="User Segments", icon=":material/groups:"
)

# -- Processed data pages --
processed_overview = st.Page(
    "pages/processed/processed_overview.py",
    title="Processed Overview",
    icon=":material/data_array:",
)
processed_tensors = st.Page(
    "pages/processed/processed_tensors.py",
    title="Tensors & Embeddings",
    icon=":material/grid_on:",
)
processed_popularity = st.Page(
    "pages/processed/processed_popularity.py",
    title="Popularity Features",
    icon=":material/local_fire_department:",
)
processed_behaviors = st.Page(
    "pages/processed/processed_behaviors.py",
    title="Behaviors & Sampling",
    icon=":material/shuffle:",
)

# -- Model analysis pages --
model_predictions = st.Page(
    "pages/model_predictions.py",
    title="Model Predictions",
    icon=":material/analytics:",
)
model_comparison = st.Page(
    "pages/model_comparison.py", title="Model Comparison", icon=":material/compare:"
)
error_analysis = st.Page(
    "pages/error_analysis.py", title="Error Analysis", icon=":material/bug_report:"
)
popularity_bias = st.Page(
    "pages/popularity_bias.py",
    title="Popularity & Bias",
    icon=":material/trending_up:",
)

# -- Navigation --
nav = st.navigation(
    {
        "Raw Dataset": [overview, news_corpus, temporal, user_behavior, user_segments],
        "Processed Data": [
            processed_overview,
            processed_tensors,
            processed_popularity,
            processed_behaviors,
        ],
        "Models": [
            model_predictions,
            model_comparison,
            error_analysis,
            popularity_bias,
        ],
    }
)

# -- Global sidebar --
with st.sidebar:
    st.markdown("# VewsX")
    st.caption("News Recommendation Visualizer")

nav.run()
