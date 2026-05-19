"""Shared model configuration dataclasses for all frameworks.

Each Config dataclass defines the hyperparameters for a specific model architecture.
These are framework-agnostic and used by Keras, PyTorch, and JAX implementations.
"""

from dataclasses import dataclass, field


@dataclass
class EncoderConfig:
    """Encoder backbone selection — describes how text becomes features.

    ``type='glove'`` keeps the legacy behaviour (pretrained GloVe word
    embeddings; models tokenize and run their own per-token encoders).
    PLM types (``bert`` | ``roberta`` | ``sbert`` | ``distilbert``)
    cache the token-level HF output at dataset-init time and expose it
    to models as a frozen lookup. Both ``text_field`` (e.g. title) and
    ``text_field_abstract`` are cached when the dataset provides the
    corresponding raw text. Models decide which views to consume.

    Model-specific concerns (poolers, view selection, etc.) live in
    each model's spec, not here.
    """

    type: str = "glove"  # 'glove' | 'bert' | 'roberta' | 'sbert' | 'distilbert'

    # GloVe-only
    embedding_size: int = 300

    # PLM-only (ignored when type=='glove')
    plm_name: str = ""  # HF model id or alias from PLM_REGISTRY
    plm_dim: int = 0  # pretrained hidden size (set at build time)
    text_field: str = "title"
    max_length: int = 32
    text_field_abstract: str = ""
    max_length_abstract: int = 50
    batch_size: int = 64

    # Trainability regime — orthogonal to the model architecture.
    #   "frozen_cached"     — encode once offline; train-time = lookup.
    #   "last_n_trainable"  — HF model in graph; freeze all but the last N
    #                         transformer blocks. (Pass 2 — not yet implemented.)
    #   "full_trainable"    — HF model in graph; everything trainable. (Pass 2.)
    trainability: str = "frozen_cached"
    trainable_layers: int = 0


@dataclass
class TextPoolerConfig:
    """Pooler over per-news token features. Used by models that consume
    token-level PLM caches and want to collapse them to a per-news
    vector at training time (e.g. NRMS's IP2-style attention pooler).

    Lives at the model level (not the encoder) because each model
    independently decides whether and how to pool token-level features.
    """

    type: str = "attention"  # 'mean' | 'cls' | 'attention' | 'gate'
    num_heads: int = 6
    head_dim: int = 128
    attention_query_dim: int = 200
    dropout_rate: float = 0.0


@dataclass
class NRMSConfig:
    """Configuration class for NRMS model parameters."""

    embedding_size: int = 300
    multiheads: int = 16
    head_dim: int = 16
    attention_hidden_dim: int = 200
    dropout_rate: float = 0.2
    seed: int = 42
    max_title_length: int = 50
    max_history_length: int = 50
    max_impressions_length: int = 5
    process_user_id: bool = False
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    # PLM-only: how to collapse the (T, plm_dim) token sequence into a
    # per-news vector. Ignored when encoder.type == 'glove'.
    text_pooler: TextPoolerConfig = field(default_factory=TextPoolerConfig)


@dataclass
class NAMLConfig:
    """Configuration class for NAML model parameters."""

    max_title_length: int = 30
    max_abstract_length: int = 50
    embedding_size: int = 300
    category_embedding_dim: int = 100
    subcategory_embedding_dim: int = 100
    cnn_filter_num: int = 400
    cnn_kernel_size: int = 3
    word_attention_query_dim: int = 200
    view_attention_query_dim: int = 200
    user_attention_query_dim: int = 200
    dropout_rate: float = 0.2
    activation: str = "relu"
    max_history_length: int = 50
    max_impressions_length: int = 5
    process_user_id: bool = False
    seed: int = 42
    encoder: EncoderConfig = field(default_factory=EncoderConfig)


@dataclass
class LSTURConfig:
    """Configuration class for LSTUR model parameters."""

    embedding_size: int = 300
    cnn_filter_num: int = 300
    cnn_kernel_size: int = 3
    cnn_activation: str = "relu"
    attention_hidden_dim: int = 200
    gru_unit: int = 300
    type: str = "ini"  # "ini" or "con" for different user encoder types
    dropout_rate: float = 0.2
    # Bernoulli mask on long-term user embeddings (paper §3.2). Distinct from
    # the regular dropout_rate so we can disable it during fine-tuning.
    user_embedding_dropout_rate: float = 0.5
    seed: int = 42
    max_title_length: int = 50
    max_history_length: int = 50
    max_impressions_length: int = 5
    process_user_id: bool = True  # LSTUR uses user ids as embedding layer
    use_category: bool = False
    use_subcategory: bool = False
    category_embedding_dim: int = 100
    subcategory_embedding_dim: int = 100
    encoder: EncoderConfig = field(default_factory=EncoderConfig)


@dataclass
class CROWNConfig:
    """Configuration for CROWN (WWW 2025).

    Defaults match the official reference implementation:
    ``reference_codes/crown-www25/config.py``.
    """

    # Word / shared embeddings
    embedding_size: int = 300
    dropout_rate: float = 0.2
    seed: int = 42

    # Category / subcategory embeddings (paper: 50 each)
    category_embedding_dim: int = 50
    subcategory_embedding_dim: int = 50

    # Transformer encoder (paper: 1 layer, 10 heads, FFN 512)
    num_heads: int = 10
    head_dim: int = 30  # 300 / 10
    feedforward_dim: int = 512
    num_layers: int = 1

    # Intent disentanglement (paper: k=3, dim=400)
    intent_num: int = 3
    intent_embedding_dim: int = 400
    attention_dim: int = 400
    alpha: float = 0.3  # auxiliary category-prediction loss weight

    # Bipartite user-news GNN (paper §3.3 eq. 8)
    # Paper uses GAT; reference code ships GraphSAGE. Both supported for ablation.
    gnn_type: str = "gat"  # 'gat' or 'graphsage'
    graph_num_layers: int = 1
    graph_hidden_dim: int = 0  # 0 = auto (news_embedding_dim); set explicitly for Keras
    gat_num_heads: int = 4
    gat_alpha: float = 0.2
    gat_concat_heads: bool = True
    # GraphSAGE-specific
    sage_aggregator: str = "mean"
    sage_normalize: bool = True

    # Candidate-aware attention (paper: scaled dot-product)
    user_attention_dim: int = 400

    # Input constraints (paper: title=32, abstract=128, history=50)
    max_title_length: int = 32
    max_abstract_length: int = 128
    max_history_length: int = 50
    max_impressions_length: int = 5

    # Training
    process_user_id: bool = False
    gradient_clip_norm: float = 4.0
    encoder: EncoderConfig = field(default_factory=EncoderConfig)


@dataclass
class GLORYConfig:
    """Configuration for GLORY (RecSys 2023).

    Global Graph-Enhanced Personalized News Recommendations.
    Defaults match the reference ``configs/small.yaml`` +
    ``configs/model/default.yaml`` from the GLORY repo.
    """

    # Word / news dimensions
    word_emb_dim: int = 300
    head_num: int = 20
    head_dim: int = 20
    attention_hidden_dim: int = 200
    dropout_rate: float = 0.2

    # Input features (matches GLORY's ``news_input`` packing:
    # [title, entity, category, subcategory, news_idx] = 30+5+1+1+1 = 38)
    title_size: int = 30
    entity_size: int = 5
    max_history_length: int = 50
    max_impressions_length: int = 5

    # Global news graph
    use_graph_type: int = 0  # 0 = trajectory, 1 = co-occurrence
    directed: bool = True
    gnn_num_layers: int = 3
    k_hops: int = 2  # subgraph sampling depth
    num_neighbors: int = 8  # max neighbors per node per hop

    # Entity support (off in v1 — add later)
    use_entity: bool = False
    entity_emb_dim: int = 100
    entity_neighbors: int = 10

    # Backend for graph ops. ``True`` swaps our pure-PyTorch
    # ``GatedGraphConv`` for ``torch_geometric.nn.GatedGraphConv``
    # (CUDA-optimized scatter/gather). Default ``False`` keeps the
    # framework-agnostic pure-PyTorch path.
    use_torchgeo: bool = False

    # Standard
    process_user_id: bool = False
    gradient_clip_norm: float = 1.0

    @property
    def news_embedding_dim(self) -> int:
        return self.head_num * self.head_dim

    @property
    def news_feature_dim(self) -> int:
        # [title | entity | category | subcategory | news_idx]
        return self.title_size + self.entity_size + 3


@dataclass
class DIGATConfig:
    """Configuration for DIGAT (EMNLP 2022 Findings).

    Dual Interactive Graph Attention Networks for news recommendation.
    Defaults match ``reference_codes/digat/config.py``.
    """

    # Word embedding
    embedding_size: int = 300
    dropout_rate: float = 0.2
    seed: int = 42

    # News encoder (MSA: multi-head self-attention)
    msa_head_num: int = 16
    msa_head_dim: int = 25  # news_embedding_dim = 16 * 25 = 400
    attention_dim: int = 256

    # Dual graph interaction
    graph_depth: int = 3

    # Semantic Augmented Graph (precomputed offline)
    sag_hops: int = 2
    sag_neighbors: int = 5
    # news_graph_size is computed: 1 + 5 + 5*4 = 26 for defaults above

    # Input constraints
    max_title_length: int = 32
    max_history_length: int = 50
    max_impressions_length: int = 5

    # Training
    process_user_id: bool = False
    gradient_clip_norm: float = 1.0
    encoder: EncoderConfig = field(default_factory=EncoderConfig)

    @property
    def news_embedding_dim(self) -> int:
        return self.msa_head_num * self.msa_head_dim

    @property
    def news_graph_size(self) -> int:
        size = 1
        neighbors = 1
        for i in range(self.sag_hops):
            if i == 0:
                neighbors *= self.sag_neighbors
            else:
                neighbors *= self.sag_neighbors - 1
            size += neighbors
        return size


@dataclass
class TCCMConfig:
    """Configuration class for TCCM model parameters.

    TCCM: Time and Content-Aware Causal Model for Unbiased News
    Recommendation (CIKM 2023). Reuses PP-Rec's news/user encoders for
    the user-content matching score and replaces the popularity branch
    with a content-aware popularity encoder driven by per-token (word /
    entity) CTR buckets and a reciprocal-power time module.
    """

    # ---- News / user encoders (shared with PP-Rec ``co1`` recipe) ----
    embedding_size: int = 300
    news_dim: int = 400
    entity_embedding_dim: int = 100
    category_embedding_dim: int = 200

    # MHSA on title words / entities.
    num_heads: int = 20
    head_dim: int = 20
    # Bidirectional MHCA between title words and entities. We reuse
    # PP-Rec's ``co1`` cross-attention layout (5 heads × 40 dim = 200
    # output, then Concat + Dense fusion) — the paper text says the same
    # 20×20 split as the self-attention, but the Add-residual variant in
    # an earlier port consistently underperformed PP-Rec ``co1`` on
    # MIND-small val/test. Sticking with the empirically-better fusion.
    co_num_heads: int = 5
    co_head_dim: int = 40
    attention_hidden_dim: int = 200

    # ---- Popularity user-modelling (history popularity embedding) ----
    popularity_embedding_bins: int = 200
    popularity_embedding_dim: int = 400

    # ---- TCCM popularity encoder (per-token bucketed CTR) ----
    pop_token_embedding_bins: int = 200
    pop_token_embedding_dim: int = 200
    pop_num_heads: int = 1
    pop_head_dim: int = 400
    pop_content_dims: tuple[int, int, int] = (256, 256, 128)

    # ---- Time / timeliness module ----
    timeliness_embedding_bins: int = 505
    timeliness_embedding_dim: int = 100
    pop_recency_dims: tuple[int, int] = (64, 64)
    timeliness_lambda: float = 2.0

    # ---- Activity gater ----
    activity_gate_dims: tuple[int, int] = (128, 64)

    # ---- Causal intervention ----
    use_causal_intervention: bool = False
    intervention_value: float = 0.5

    # ---- Feature flags ----
    use_entity: bool = True
    use_activity_gate: bool = True

    # ---- Training ----
    dropout_rate: float = 0.2
    seed: int = 42

    # ---- Input constraints ----
    max_title_length: int = 30
    max_history_length: int = 50
    max_impressions_length: int = 5
    max_entities: int = 5

    process_user_id: bool = False
    encoder: EncoderConfig = field(default_factory=EncoderConfig)


@dataclass
class PPRecConfig:
    """Configuration class for PP-Rec model parameters.

    PP-Rec: News Recommendation with Personalized User Interest and
    Time-aware News Popularity (ACL 2021).
    """

    # Embedding dimensions
    embedding_size: int = 300
    news_dim: int = 400
    entity_embedding_dim: int = 100
    category_embedding_dim: int = 200

    # Multi-head self-attention (title MHSA, entity MHSA)
    num_heads: int = 20
    head_dim: int = 20
    # Cross-attention between title words and entities (paper "co1" encoder).
    # Paper uses 5 heads x 40 dim = 200 total.
    co_num_heads: int = 5
    co_head_dim: int = 40
    attention_hidden_dim: int = 200

    # Popularity embeddings
    popularity_embedding_bins: int = 200
    popularity_embedding_dim: int = 400
    recency_embedding_bins: int = 1500
    recency_embedding_dim: int = 100
    ctr_scaler_init: float = 19.0

    # PopularityPredictor MLP widths (paper PP-Rec official code).
    # content scorer: news_dim -> dims[0] -> dims[1] -> dims[2] -> 1
    pop_content_dims: tuple[int, int, int] = (256, 256, 128)
    # recency scorer: recency_emb_dim -> dims[0] -> dims[1] -> 1
    pop_recency_dims: tuple[int, int] = (64, 64)
    # gate MLP: concat[news, recency_emb] -> dims[0] -> dims[1] -> 1 (sigmoid)
    pop_gate_dims: tuple[int, int] = (128, 64)
    # ActivityGater hidden width: news_dim -> hidden -> 1 (sigmoid)
    activity_gate_hidden_dim: int = 64

    # Feature flags
    use_entity: bool = True
    use_recency: bool = True
    use_ctr: bool = True
    use_activity_gate: bool = True

    # Training
    dropout_rate: float = 0.2
    seed: int = 42

    # Input constraints
    max_title_length: int = 32
    max_history_length: int = 50
    max_impressions_length: int = 5
    max_entities: int = 5

    process_user_id: bool = False
    encoder: EncoderConfig = field(default_factory=EncoderConfig)


@dataclass
class CAUMConfig:
    """Configuration for CAUM (SIGIR 2022).

    News Recommendation with Candidate-aware User Modeling.
    Defaults match the official reference implementation.
    """

    # Word embeddings
    embedding_size: int = 300
    dropout_rate: float = 0.2
    seed: int = 42

    # News encoder — multi-head self-attention over title words
    news_num_heads: int = 20
    news_head_dim: int = 20
    news_attention_hidden_dim: int = 200

    # Entity encoder (reference: 4 heads × 40 dim = 160)
    entity_embedding_dim: int = 100
    entity_num_heads: int = 4
    entity_head_dim: int = 40

    # Category encoder (reference: embedding 100d → Dense 100d)
    category_embedding_dim: int = 100

    # Feature flags
    use_entity: bool = True
    use_category: bool = True

    # Candidate-aware self-attention (Candi-SelfAtt)
    candi_selfatt_num_heads: int = 20
    candi_selfatt_head_dim: int = 20

    # Candidate-aware CNN (Candi-CNN) — window = 2*half_window+1
    candi_cnn_half_window: int = 1

    # Candidate-aware attention (Candi-Att) — DNN scorer
    candi_att_hidden_dim: int = 400
    candi_att_mid_dim: int = 256

    # Output news/user dimension after fusion
    news_dim: int = 400

    # Input constraints
    max_title_length: int = 30
    max_history_length: int = 50
    max_impressions_length: int = 5
    max_entities: int = 5

    # Training
    process_user_id: bool = False
    encoder: EncoderConfig = field(default_factory=EncoderConfig)


@dataclass
class MINERConfig:
    """Configuration for MINER (ACL Findings 2022).

    Multi-Interest Matching Network for News Recommendation.
    Defaults match the paper's reported hyperparameters.
    """

    # Word embeddings
    embedding_size: int = 300
    dropout_rate: float = 0.2
    seed: int = 42

    # News encoder (multi-head self-attention)
    num_heads: int = 20
    head_dim: int = 15
    attention_hidden_dim: int = 200

    # Poly attention (multi-interest)
    num_interest_vectors: int = 32  # K in the paper
    context_code_dim: int = 200

    # Disagreement regularization
    disagreement_beta: float = 0.0  # 0.8 in paper (with BERT)

    # Input constraints
    max_title_length: int = 32
    max_history_length: int = 50
    max_impressions_length: int = 5

    # Training
    process_user_id: bool = False
