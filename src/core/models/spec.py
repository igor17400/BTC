"""Spec-to-config translation layer.

Converts YAML DSL spec dicts into Config dataclasses that model constructors accept,
and provides a factory for building models from specs.
"""

import importlib
from typing import Any

from omegaconf import DictConfig

from .configs import (
    CAUMConfig,
    CROWNConfig,
    DIGATConfig,
    EncoderConfig,
    GLORYConfig,
    LSTURConfig,
    MINERConfig,
    NAMLConfig,
    NRMSConfig,
    PPRecConfig,
    TCCMConfig,
    TextPoolerConfig,
)

# ---------------------------------------------------------------------------
# Spec → Config translation functions
# ---------------------------------------------------------------------------


def _encoder_from_cfg(encoder_cfg: DictConfig | None) -> EncoderConfig:
    """Build an EncoderConfig from the top-level ``cfg.encoder`` dict.

    Accepts ``None`` (defaults to GloVe) so the existing call-sites that
    don't yet pass an encoder keep working.
    """
    if encoder_cfg is None:
        return EncoderConfig()
    return EncoderConfig(
        type=encoder_cfg.get("type", "glove"),
        embedding_size=encoder_cfg.get("embedding_size", 300),
        plm_name=encoder_cfg.get("plm_name", ""),
        plm_dim=encoder_cfg.get("plm_dim", 0),
        text_field=encoder_cfg.get("text_field", "title"),
        max_length=encoder_cfg.get("max_length", 32),
        text_field_abstract=encoder_cfg.get("text_field_abstract", "") or "",
        max_length_abstract=encoder_cfg.get("max_length_abstract", 50),
        batch_size=encoder_cfg.get("batch_size", 64),
        trainability=encoder_cfg.get("trainability", "frozen_cached"),
        trainable_layers=encoder_cfg.get("trainable_layers", 0),
    )


def _text_pooler_from_arch(arch: DictConfig) -> TextPoolerConfig:
    """Read the ``text_pooler`` block from a model architecture spec.

    Tolerates missing sub-fields so old specs (no ``text_pooler`` block)
    pick up the dataclass defaults.
    """
    pooler_cfg = arch.get("text_pooler")
    if pooler_cfg is None:
        return TextPoolerConfig()
    return TextPoolerConfig(
        type=pooler_cfg.get("type", "attention"),
        num_heads=pooler_cfg.get("num_heads", 6),
        head_dim=pooler_cfg.get("head_dim", 128),
        attention_query_dim=pooler_cfg.get("attention_query_dim", 200),
        dropout_rate=pooler_cfg.get("dropout_rate", 0.0),
    )


def spec_to_nrms_config(
    spec: DictConfig, encoder: DictConfig | None = None
) -> NRMSConfig:
    """Convert a parsed NRMS spec into NRMSConfig."""
    arch = spec.model.architecture
    return NRMSConfig(
        embedding_size=spec.model.embedding.size,
        news_num_heads=arch.news_encoder.num_heads,
        user_num_heads=arch.user_encoder.num_heads,
        head_dim=arch.news_encoder.head_dim,
        attention_hidden_dim=arch.news_encoder.attention_hidden_dim,
        dropout_rate=spec.model.dropout_rate,
        seed=spec.model.seed,
        max_title_length=spec.inputs.title.max_length,
        max_history_length=spec.inputs.history.max_length,
        max_impressions_length=spec.inputs.impressions.max_length,
        process_user_id=spec.inputs.get("process_user_id", False),
        encoder=_encoder_from_cfg(encoder),
    )


def spec_to_naml_config(
    spec: DictConfig, encoder: DictConfig | None = None
) -> NAMLConfig:
    """Convert a parsed NAML spec into NAMLConfig."""
    arch = spec.model.architecture
    views = arch.news_encoder.views
    return NAMLConfig(
        max_title_length=spec.inputs.title.max_length,
        max_abstract_length=spec.inputs.abstract.max_length,
        embedding_size=spec.model.embedding.size,
        category_embedding_dim=views.category.embedding_dim,
        subcategory_embedding_dim=views.subcategory.embedding_dim,
        cnn_filter_num=views.title.filter_num,
        cnn_kernel_size=views.title.kernel_size,
        word_attention_query_dim=views.title.attention_query_dim,
        view_attention_query_dim=arch.news_encoder.view_attention_query_dim,
        user_attention_query_dim=arch.user_encoder.attention_query_dim,
        dropout_rate=spec.model.dropout_rate,
        activation=views.title.activation,
        max_history_length=spec.inputs.history.max_length,
        max_impressions_length=spec.inputs.impressions.max_length,
        process_user_id=spec.inputs.get("process_user_id", False),
        seed=spec.model.seed,
        encoder=_encoder_from_cfg(encoder),
    )


def spec_to_lstur_config(
    spec: DictConfig, encoder: DictConfig | None = None
) -> LSTURConfig:
    """Convert a parsed LSTUR spec into LSTURConfig."""
    arch = spec.model.architecture
    return LSTURConfig(
        embedding_size=spec.model.embedding.size,
        cnn_filter_num=arch.news_encoder.filter_num,
        cnn_kernel_size=arch.news_encoder.kernel_size,
        cnn_activation=arch.news_encoder.activation,
        attention_hidden_dim=arch.news_encoder.attention_hidden_dim,
        gru_unit=arch.user_encoder.gru_unit,
        type=arch.user_encoder.variant,
        dropout_rate=spec.model.dropout_rate,
        user_embedding_dropout_rate=spec.model.get("user_embedding_dropout_rate", 0.5),
        seed=spec.model.seed,
        max_title_length=spec.inputs.title.max_length,
        max_history_length=spec.inputs.history.max_length,
        max_impressions_length=spec.inputs.impressions.max_length,
        process_user_id=spec.inputs.get("process_user_id", True),
        use_category=spec.inputs.get("process_category", False),
        use_subcategory=spec.inputs.get("process_subcategory", False),
        category_embedding_dim=spec.model.get("category_embedding_dim", 100),
        subcategory_embedding_dim=spec.model.get("subcategory_embedding_dim", 100),
        encoder=_encoder_from_cfg(encoder),
    )


def spec_to_crown_config(
    spec: DictConfig, encoder: DictConfig | None = None
) -> CROWNConfig:
    """Convert a parsed CROWN spec into CROWNConfig."""
    arch = spec.model.architecture
    ne = arch.news_encoder
    ue = arch.user_encoder
    return CROWNConfig(
        embedding_size=spec.model.embedding.size,
        dropout_rate=spec.model.dropout_rate,
        seed=spec.model.seed,
        # Category embeddings
        category_embedding_dim=spec.model.get("category_embedding_dim", 50),
        subcategory_embedding_dim=spec.model.get("subcategory_embedding_dim", 50),
        # Transformer
        num_heads=ne.num_heads,
        head_dim=ne.head_dim,
        feedforward_dim=ne.feedforward_dim,
        num_layers=ne.num_layers,
        # Intent disentanglement
        intent_num=ne.intent_num,
        intent_embedding_dim=ne.intent_embedding_dim,
        attention_dim=ne.attention_dim,
        alpha=spec.training.get("auxiliary_alpha", 0.3),
        # Bipartite GNN (paper §3.3)
        gnn_type=ue.get("gnn_type", "gat"),
        graph_num_layers=ue.get("graph_num_layers", 1),
        graph_hidden_dim=ue.get("graph_hidden_dim", 0),
        gat_num_heads=ue.get("gat_num_heads", 4),
        gat_alpha=ue.get("gat_alpha", 0.2),
        gat_concat_heads=ue.get("gat_concat_heads", True),
        sage_aggregator=ue.get("sage_aggregator", "mean"),
        sage_normalize=ue.get("sage_normalize", True),
        # Candidate-aware attention
        user_attention_dim=ue.get("user_attention_dim", 400),
        # Input constraints
        max_title_length=spec.inputs.title.max_length,
        max_abstract_length=spec.inputs.abstract.max_length,
        max_history_length=spec.inputs.history.max_length,
        max_impressions_length=spec.inputs.impressions.max_length,
        process_user_id=spec.inputs.get("process_user_id", False),
        encoder=_encoder_from_cfg(encoder),
    )


# ---------------------------------------------------------------------------
# Spec → Config dispatcher
# ---------------------------------------------------------------------------


def spec_to_pprec_config(
    spec: DictConfig, encoder: DictConfig | None = None
) -> PPRecConfig:
    """Convert a parsed PP-Rec spec into PPRecConfig."""
    arch = spec.model.architecture
    ne = arch.news_encoder
    return PPRecConfig(
        embedding_size=spec.model.embedding.size,
        news_dim=ne.get("news_dim", 400),
        entity_embedding_dim=ne.get("entity_embedding_dim", 100),
        category_embedding_dim=ne.get("category_embedding_dim", 200),
        num_heads=ne.num_heads,
        head_dim=ne.head_dim,
        co_num_heads=ne.get("co_num_heads", 5),
        co_head_dim=ne.get("co_head_dim", 40),
        attention_hidden_dim=ne.attention_hidden_dim,
        popularity_embedding_bins=arch.get("popularity_embedding_bins", 200),
        popularity_embedding_dim=arch.get("popularity_embedding_dim", 400),
        recency_embedding_bins=arch.get("recency_embedding_bins", 1500),
        recency_embedding_dim=arch.get("recency_embedding_dim", 100),
        ctr_scaler_init=arch.get("ctr_scaler_init", 19.0),
        pop_content_dims=tuple(arch.get("pop_content_dims", (256, 256, 128))),
        pop_recency_dims=tuple(arch.get("pop_recency_dims", (64, 64))),
        pop_gate_dims=tuple(arch.get("pop_gate_dims", (128, 64))),
        activity_gate_hidden_dim=arch.get("activity_gate_hidden_dim", 64),
        use_entity=arch.get("use_entity", True),
        use_recency=arch.get("use_recency", True),
        use_ctr=arch.get("use_ctr", True),
        use_activity_gate=arch.get("use_activity_gate", True),
        dropout_rate=spec.model.dropout_rate,
        seed=spec.model.seed,
        max_title_length=spec.inputs.title.max_length,
        max_history_length=spec.inputs.history.max_length,
        max_impressions_length=spec.inputs.impressions.max_length,
        max_entities=spec.inputs.get("max_entities", 5),
        process_user_id=spec.inputs.get("process_user_id", False),
        encoder=_encoder_from_cfg(encoder),
    )


def spec_to_digat_config(
    spec: DictConfig, encoder: DictConfig | None = None
) -> DIGATConfig:
    """Convert a parsed DIGAT spec into DIGATConfig."""

    ne = spec.model.architecture.news_encoder
    ge = spec.model.architecture.graph_encoder
    return DIGATConfig(
        embedding_size=spec.model.embedding.size,
        dropout_rate=spec.model.dropout_rate,
        seed=spec.model.seed,
        msa_head_num=ne.get("msa_head_num", 16),
        msa_head_dim=ne.get("msa_head_dim", 25),
        attention_dim=ne.get("attention_dim", 256),
        graph_depth=ge.get("graph_depth", 3),
        sag_hops=ge.get("sag_hops", 2),
        sag_neighbors=ge.get("sag_neighbors", 5),
        max_title_length=spec.inputs.title.max_length,
        max_history_length=spec.inputs.history.max_length,
        max_impressions_length=spec.inputs.impressions.max_length,
        process_user_id=spec.inputs.get("process_user_id", False),
        encoder=_encoder_from_cfg(encoder),
    )


def spec_to_glory_config(spec: DictConfig) -> GLORYConfig:
    """Convert a parsed GLORY spec into GLORYConfig."""
    ne = spec.model.architecture.news_encoder
    ge = spec.model.architecture.graph_encoder
    return GLORYConfig(
        word_emb_dim=spec.model.embedding.size,
        head_num=ne.get("head_num", 20),
        head_dim=ne.get("head_dim", 20),
        attention_hidden_dim=ne.get("attention_hidden_dim", 200),
        dropout_rate=spec.model.dropout_rate,
        title_size=spec.inputs.title.max_length,
        entity_size=spec.inputs.get("entity", {}).get("max_length", 5),
        max_history_length=spec.inputs.history.max_length,
        max_impressions_length=spec.inputs.impressions.max_length,
        use_graph_type=ge.get("use_graph_type", 0),
        directed=ge.get("directed", True),
        gnn_num_layers=ge.get("gnn_num_layers", 3),
        k_hops=ge.get("k_hops", 2),
        num_neighbors=ge.get("num_neighbors", 8),
        use_entity=spec.model.get("use_entity", False),
        use_torchgeo=spec.model.get("use_torchgeo", False),
        process_user_id=spec.inputs.get("process_user_id", False),
    )


def spec_to_tccm_config(
    spec: DictConfig, encoder: DictConfig | None = None
) -> TCCMConfig:
    """Convert a parsed TCCM spec into TCCMConfig."""
    arch = spec.model.architecture
    ne = arch.news_encoder
    pop = arch.popularity_encoder
    time_module = arch.time_module
    return TCCMConfig(
        embedding_size=spec.model.embedding.size,
        news_dim=ne.get("news_dim", 400),
        entity_embedding_dim=ne.get("entity_embedding_dim", 100),
        category_embedding_dim=ne.get("category_embedding_dim", 200),
        num_heads=ne.num_heads,
        head_dim=ne.head_dim,
        co_num_heads=ne.get("co_num_heads", 5),
        co_head_dim=ne.get("co_head_dim", 40),
        attention_hidden_dim=ne.attention_hidden_dim,
        popularity_embedding_bins=arch.get("popularity_embedding_bins", 200),
        popularity_embedding_dim=arch.get("popularity_embedding_dim", 400),
        pop_token_embedding_bins=pop.get("token_embedding_bins", 200),
        pop_token_embedding_dim=pop.get("token_embedding_dim", 200),
        pop_num_heads=pop.get("num_heads", 1),
        pop_head_dim=pop.get("head_dim", 400),
        pop_content_dims=tuple(pop.get("content_dims", (256, 256, 128))),
        timeliness_embedding_bins=time_module.get("embedding_bins", 505),
        timeliness_embedding_dim=time_module.get("embedding_dim", 100),
        pop_recency_dims=tuple(time_module.get("dense_dims", (64, 64))),
        timeliness_lambda=time_module.get("lambda_exp", 2.0),
        activity_gate_dims=tuple(arch.get("activity_gate_dims", (128, 64))),
        use_causal_intervention=arch.get("use_causal_intervention", False),
        intervention_value=arch.get("intervention_value", 0.5),
        use_entity=arch.get("use_entity", True),
        use_activity_gate=arch.get("use_activity_gate", True),
        dropout_rate=spec.model.dropout_rate,
        seed=spec.model.seed,
        max_title_length=spec.inputs.title.max_length,
        max_history_length=spec.inputs.history.max_length,
        max_impressions_length=spec.inputs.impressions.max_length,
        max_entities=spec.inputs.get("max_entities", 5),
        process_user_id=spec.inputs.get("process_user_id", False),
        encoder=_encoder_from_cfg(encoder),
    )


def spec_to_miner_config(spec: DictConfig) -> MINERConfig:
    """Convert a parsed MINER spec into MINERConfig."""
    arch = spec.model.architecture
    ne = arch.news_encoder
    ue = arch.user_encoder
    return MINERConfig(
        embedding_size=spec.model.embedding.size,
        dropout_rate=spec.model.dropout_rate,
        seed=spec.model.seed,
        num_heads=ne.num_heads,
        head_dim=ne.head_dim,
        attention_hidden_dim=ne.attention_hidden_dim,
        num_interest_vectors=ue.get("num_interest_vectors", 32),
        context_code_dim=ue.get("context_code_dim", 200),
        disagreement_beta=spec.training.get("disagreement_beta", 0.8),
        max_title_length=spec.inputs.title.max_length,
        max_history_length=spec.inputs.history.max_length,
        max_impressions_length=spec.inputs.impressions.max_length,
        process_user_id=spec.inputs.get("process_user_id", False),
    )


def spec_to_caum_config(
    spec: DictConfig, encoder: DictConfig | None = None
) -> CAUMConfig:
    """Convert a parsed CAUM spec into CAUMConfig."""
    arch = spec.model.architecture
    ne = arch.news_encoder
    ue = arch.user_encoder
    return CAUMConfig(
        embedding_size=spec.model.embedding.size,
        dropout_rate=spec.model.dropout_rate,
        seed=spec.model.seed,
        news_num_heads=ne.num_heads,
        news_head_dim=ne.head_dim,
        news_attention_hidden_dim=ne.attention_hidden_dim,
        entity_embedding_dim=ne.get("entity_embedding_dim", 100),
        entity_num_heads=ne.get("entity_num_heads", 4),
        entity_head_dim=ne.get("entity_head_dim", 40),
        category_embedding_dim=ne.get("category_embedding_dim", 100),
        use_entity=spec.model.get("use_entity", True),
        use_category=spec.model.get("use_category", True),
        candi_selfatt_num_heads=ue.candi_selfatt.num_heads,
        candi_selfatt_head_dim=ue.candi_selfatt.head_dim,
        candi_cnn_half_window=ue.candi_cnn.half_window,
        candi_att_hidden_dim=ue.candi_att.hidden_dim,
        candi_att_mid_dim=ue.candi_att.mid_dim,
        news_dim=spec.model.get("news_dim", 400),
        max_title_length=spec.inputs.title.max_length,
        max_history_length=spec.inputs.history.max_length,
        max_impressions_length=spec.inputs.impressions.max_length,
        max_entities=spec.inputs.get("max_entities", 5),
        process_user_id=spec.inputs.get("process_user_id", False),
        encoder=_encoder_from_cfg(encoder),
    )


_SPEC_CONVERTERS = {
    "caum": spec_to_caum_config,
    "nrms": spec_to_nrms_config,
    "naml": spec_to_naml_config,
    "lstur": spec_to_lstur_config,
    "crown": spec_to_crown_config,
    "digat": spec_to_digat_config,
    "glory": spec_to_glory_config,
    "pprec": spec_to_pprec_config,
    "miner": spec_to_miner_config,
    "tccm": spec_to_tccm_config,
}


def spec_to_config(spec: DictConfig, encoder: DictConfig | None = None):
    """Convert a YAML spec to the appropriate Config dataclass.

    Args:
        spec: Parsed YAML spec (the `spec` key from the Hydra config).
        encoder: Optional top-level encoder config (``cfg.encoder``). Only
            consumed by converters that have been updated to accept it
            (currently NRMS); silently ignored by others.

    Returns:
        The corresponding Config dataclass instance.
    """
    model_name = spec.model.name.lower()
    converter = _SPEC_CONVERTERS.get(model_name)
    if converter is None:
        raise ValueError(
            f"Unknown model '{model_name}' in spec. "
            f"Supported: {list(_SPEC_CONVERTERS.keys())}"
        )
    # Try the new (spec, encoder) signature first; fall back to legacy.
    try:
        return converter(spec, encoder=encoder)
    except TypeError:
        return converter(spec)


# ---------------------------------------------------------------------------
# Model class resolver
# ---------------------------------------------------------------------------

_MODEL_CLASS_PATHS = {
    "keras": {
        "caum": "src.frameworks.keras.models.caum.CAUM",
        "nrms": "src.frameworks.keras.models.nrms.NRMS",
        "naml": "src.frameworks.keras.models.naml.NAML",
        "lstur": "src.frameworks.keras.models.lstur.LSTUR",
        "crown": "src.frameworks.keras.models.crown.CROWN",
        "digat": "src.frameworks.keras.models.digat.DIGAT",
        "pprec": "src.frameworks.keras.models.pprec.PPRec",
        "glory": "src.frameworks.keras.models.glory.GLORY",
        "miner": "src.frameworks.keras.models.miner.MINER",
        "tccm": "src.frameworks.keras.models.tccm.TCCM",
    },
    "pytorch": {
        "caum": "src.frameworks.pytorch.models.caum.model.CAUM",
        "nrms": "src.frameworks.pytorch.models.nrms.model.NRMS",
        "naml": "src.frameworks.pytorch.models.naml.model.NAML",
        "lstur": "src.frameworks.pytorch.models.lstur.model.LSTUR",
        "crown": "src.frameworks.pytorch.models.crown.CROWN",
        "digat": "src.frameworks.pytorch.models.digat.DIGAT",
        "glory": "src.frameworks.pytorch.models.glory.GLORY",
        "pprec": "src.frameworks.pytorch.models.pprec.PPRec",
        "miner": "src.frameworks.pytorch.models.miner.MINER",
        "tccm": "src.frameworks.pytorch.models.tccm.TCCM",
    },
    "jax": {
        "caum": "src.frameworks.jax.models.caum.model.CAUM",
        "nrms": "src.frameworks.jax.models.nrms.model.NRMS",
        "naml": "src.frameworks.jax.models.naml.model.NAML",
        "lstur": "src.frameworks.jax.models.lstur.model.LSTUR",
        "crown": "src.frameworks.jax.models.crown.CROWN",
        "pprec": "src.frameworks.jax.models.pprec.PPRec",
        "digat": "src.frameworks.jax.models.digat.DIGAT",
        "glory": "src.frameworks.jax.models.glory.GLORY",
        "miner": "src.frameworks.jax.models.miner.MINER",
        "tccm": "src.frameworks.jax.models.tccm.TCCM",
    },
}


def get_model_class(model_name: str, framework: str):
    """Resolve the model class for a given model name and framework.

    Args:
        model_name: Model name (e.g., "nrms", "naml").
        framework: Framework name (e.g., "keras", "pytorch", "jax").

    Returns:
        The model class.
    """
    framework = framework.lower()
    model_name = model_name.lower()

    framework_models = _MODEL_CLASS_PATHS.get(framework)
    if framework_models is None:
        raise ValueError(
            f"Unknown framework '{framework}'. Supported: {list(_MODEL_CLASS_PATHS.keys())}"
        )

    class_path = framework_models.get(model_name)
    if class_path is None:
        raise ValueError(
            f"Model '{model_name}' not available for framework '{framework}'. "
            f"Available: {list(framework_models.keys())}"
        )

    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


# ---------------------------------------------------------------------------
# Top-level factory
# ---------------------------------------------------------------------------


def build_model_from_spec(
    spec: DictConfig,
    framework: str,
    processed_news: dict[str, Any],
    encoder: DictConfig | None = None,
    **extra_kwargs,
):
    """Build a model from a YAML spec.

    Args:
        spec: Parsed YAML spec (the `spec` key from the Hydra config).
        framework: Target framework ("keras", "pytorch", "jax").
        processed_news: Processed news data dict (vocab_size, embeddings, etc.).
        encoder: Optional top-level encoder config (``cfg.encoder``) — when
            provided and the model supports it, swaps the GloVe news encoder
            for a frozen PLM lookup.
        **extra_kwargs: Additional kwargs passed to the model constructor.

    Returns:
        An instantiated model.
    """
    model_name = spec.model.name.lower()
    config = spec_to_config(spec, encoder=encoder)
    model_class = get_model_class(model_name, framework)

    # Build kwargs from config fields + processed_news
    kwargs = {field: getattr(config, field) for field in config.__dataclass_fields__}
    kwargs["processed_news"] = processed_news
    kwargs.update(extra_kwargs)

    return model_class(**kwargs)
