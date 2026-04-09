"""Graph and transformer layers for Keras (CROWN model).

Layers used exclusively by the CROWN news recommendation model:
- :class:`GraphSAGELayer` — bipartite message-passing.
- :class:`MultiHeadAttentionBlock` — MAB with residual + layer norm.
- :class:`GraphAttentionLayer` — bipartite GAT.
"""

import keras
from keras import layers, ops

class GraphSAGELayer(layers.Layer):
    """GraphSAGE layer implementation for CROWN paper.

    This layer implements GraphSAGE specifically for the bipartite user-news graph
    used in the CROWN model, with mutual updates between user and news embeddings.

    Args:
        units: Output dimension for both user and news embeddings
        aggregator: Type of aggregation ('mean', 'max', 'sum', 'attention')
        dropout_rate: Dropout rate
        activation: Activation function
        seed: Random seed
        normalize: Whether to L2-normalize the output embeddings
    """

    def __init__(
        self,
        units,
        aggregator="mean",
        dropout_rate=0.0,
        activation="relu",
        seed=42,
        normalize=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.aggregator = aggregator
        self.dropout_rate = dropout_rate
        self.activation = keras.activations.get(activation)
        self.seed = seed
        self.normalize = normalize

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                "GraphSAGELayer expects 3 inputs: (user_features, news_features, adjacency_matrix)"
            )

        user_features_shape, news_features_shape, _ = input_shape
        user_dim = user_features_shape[-1]
        news_dim = news_features_shape[-1]

        self.W_user_self = self.add_weight(
            name="W_user_self",
            shape=(user_dim, self.units),
            initializer=keras.initializers.GlorotUniform(seed=self.seed),
            trainable=True,
        )
        self.W_user_neigh = self.add_weight(
            name="W_user_neigh",
            shape=(news_dim, self.units),
            initializer=keras.initializers.GlorotUniform(seed=self.seed),
            trainable=True,
        )
        self.b_user = self.add_weight(
            name="b_user",
            shape=(self.units,),
            initializer=keras.initializers.Zeros(),
            trainable=True,
        )
        self.W_news_self = self.add_weight(
            name="W_news_self",
            shape=(news_dim, self.units),
            initializer=keras.initializers.GlorotUniform(seed=self.seed),
            trainable=True,
        )
        self.W_news_neigh = self.add_weight(
            name="W_news_neigh",
            shape=(user_dim, self.units),
            initializer=keras.initializers.GlorotUniform(seed=self.seed),
            trainable=True,
        )
        self.b_news = self.add_weight(
            name="b_news",
            shape=(self.units,),
            initializer=keras.initializers.Zeros(),
            trainable=True,
        )
        if self.aggregator == "attention":
            self.W_att = self.add_weight(
                name="W_att",
                shape=(self.units * 2, 1),
                initializer=keras.initializers.GlorotUniform(seed=self.seed),
                trainable=True,
            )
        self.dropout = layers.Dropout(self.dropout_rate, seed=self.seed)
        super().build(input_shape)

    def aggregate_neighbors(self, node_features, neighbor_features, adjacency_weights):
        if self.aggregator == "mean":
            neighbor_sum = ops.matmul(adjacency_weights, neighbor_features)
            degree = ops.sum(adjacency_weights, axis=-1, keepdims=True)
            degree = ops.maximum(degree, keras.backend.epsilon())
            return neighbor_sum / degree
        elif self.aggregator == "max":
            expanded_neighbors = ops.expand_dims(neighbor_features, axis=1)
            expanded_adj = ops.expand_dims(adjacency_weights, axis=-1)
            masked_features = expanded_neighbors * expanded_adj
            mask = ops.expand_dims(adjacency_weights, axis=-1) > 0
            masked_features = ops.where(mask, masked_features, -1e9)
            return ops.max(masked_features, axis=2)
        elif self.aggregator == "sum":
            return ops.matmul(adjacency_weights, neighbor_features)
        elif self.aggregator == "attention":
            num_nodes = ops.shape(node_features)[1]
            num_neighbors = ops.shape(neighbor_features)[1]
            node_exp = ops.expand_dims(node_features, axis=2)
            neighbor_exp = ops.expand_dims(neighbor_features, axis=1)
            node_broadcast = ops.broadcast_to(
                node_exp,
                (
                    ops.shape(node_features)[0],
                    num_nodes,
                    num_neighbors,
                    ops.shape(node_features)[2],
                ),
            )
            neighbor_broadcast = ops.broadcast_to(
                neighbor_exp,
                (
                    ops.shape(neighbor_features)[0],
                    num_nodes,
                    num_neighbors,
                    ops.shape(neighbor_features)[2],
                ),
            )
            concat_features = ops.concatenate(
                [node_broadcast, neighbor_broadcast], axis=-1
            )
            attention_scores = ops.squeeze(
                ops.matmul(concat_features, self.W_att), axis=-1
            )
            masked_scores = ops.where(adjacency_weights > 0, attention_scores, -1e9)
            attention_weights = ops.softmax(masked_scores, axis=-1)
            return ops.matmul(attention_weights, neighbor_features)
        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator}")

    def call(self, inputs, training=None):
        user_features, news_features, adjacency_matrix = inputs
        ops.shape(user_features)[0]
        num_users = ops.shape(user_features)[1]
        ops.shape(news_features)[1]

        user_to_news = adjacency_matrix[:, :num_users, num_users:]
        news_to_user = adjacency_matrix[:, num_users:, :num_users]

        news_for_users = self.aggregate_neighbors(
            user_features, news_features, user_to_news
        )
        user_self = ops.matmul(user_features, self.W_user_self)
        user_neigh = ops.matmul(news_for_users, self.W_user_neigh)
        updated_users = self.activation(user_self + user_neigh + self.b_user)
        updated_users = self.dropout(updated_users, training=training)

        users_for_news = self.aggregate_neighbors(
            news_features, user_features, news_to_user
        )
        news_self = ops.matmul(news_features, self.W_news_self)
        news_neigh = ops.matmul(users_for_news, self.W_news_neigh)
        updated_news = self.activation(news_self + news_neigh + self.b_news)
        updated_news = self.dropout(updated_news, training=training)

        if self.normalize:
            user_norm = ops.sqrt(
                ops.sum(ops.square(updated_users), axis=-1, keepdims=True)
            )
            updated_users = updated_users / (user_norm + keras.backend.epsilon())
            news_norm = ops.sqrt(
                ops.sum(ops.square(updated_news), axis=-1, keepdims=True)
            )
            updated_news = updated_news / (news_norm + keras.backend.epsilon())

        return updated_users, updated_news

    def compute_output_shape(self, input_shape):
        user_shape, news_shape, _ = input_shape
        return (
            (user_shape[0], user_shape[1], self.units),
            (news_shape[0], news_shape[1], self.units),
        )


class MultiHeadAttentionBlock(layers.Layer):
    """Multi-head attention block (MAB) for set-based operations.

    Args:
        dim_out: Output dimension
        num_heads: Number of attention heads
        use_layer_norm: Whether to use layer normalization
    """

    def __init__(self, dim_out, num_heads, use_layer_norm=True, seed=42, **kwargs):
        super().__init__(**kwargs)
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.use_layer_norm = use_layer_norm
        self.seed = seed

    def build(self, input_shape):
        if isinstance(input_shape, tuple) and len(input_shape) == 2:
            q_shape, k_shape = input_shape
        else:
            pass

        self.fc_q = layers.Dense(
            self.dim_out,
            use_bias=True,
            kernel_initializer=keras.initializers.GlorotUniform(seed=self.seed),
        )
        self.fc_k = layers.Dense(
            self.dim_out,
            use_bias=True,
            kernel_initializer=keras.initializers.GlorotUniform(seed=self.seed),
        )
        self.fc_v = layers.Dense(
            self.dim_out,
            use_bias=True,
            kernel_initializer=keras.initializers.GlorotUniform(seed=self.seed),
        )
        self.fc_o = layers.Dense(
            self.dim_out,
            use_bias=True,
            kernel_initializer=keras.initializers.GlorotUniform(seed=self.seed),
        )
        if self.use_layer_norm:
            self.ln0 = layers.LayerNormalization()
            self.ln1 = layers.LayerNormalization()
        self.dropout = layers.Dropout(0.1)
        super().build(input_shape)

    def call(self, inputs, training=None):
        if isinstance(inputs, tuple):
            Q, K = inputs
        else:
            Q = K = inputs

        Q_proj = self.fc_q(Q)
        K_proj = self.fc_k(K)
        V_proj = self.fc_v(K)

        q_shape = ops.shape(Q)
        k_shape = ops.shape(K)
        batch_size = q_shape[0]
        seq_len_q = q_shape[1]
        seq_len_k = k_shape[1]
        head_dim = self.dim_out // self.num_heads

        Q_proj = ops.reshape(Q_proj, (batch_size, seq_len_q, self.num_heads, head_dim))
        K_proj = ops.reshape(K_proj, (batch_size, seq_len_k, self.num_heads, head_dim))
        V_proj = ops.reshape(V_proj, (batch_size, seq_len_k, self.num_heads, head_dim))

        Q_proj = ops.transpose(Q_proj, (0, 2, 1, 3))
        K_proj = ops.transpose(K_proj, (0, 2, 1, 3))
        V_proj = ops.transpose(V_proj, (0, 2, 1, 3))

        scores = ops.matmul(Q_proj, ops.transpose(K_proj, (0, 1, 3, 2)))
        scores = scores / ops.sqrt(ops.cast(head_dim, scores.dtype))
        attention_weights = ops.softmax(scores, axis=-1)
        attention_weights = self.dropout(attention_weights, training=training)

        attention_output = ops.matmul(attention_weights, V_proj)
        attention_output = ops.transpose(attention_output, (0, 2, 1, 3))
        attention_output = ops.reshape(
            attention_output, (batch_size, seq_len_q, self.dim_out)
        )

        out = self.fc_o(attention_output)
        out = self.dropout(out, training=training)
        out = out + Q
        if self.use_layer_norm:
            out = self.ln0(out)

        ff = self.fc_o(ops.relu(out))
        out = out + ff
        if self.use_layer_norm:
            out = self.ln1(out)

        return out

    def compute_output_shape(self, input_shape):
        return input_shape


class GraphAttentionLayer(layers.Layer):
    """Graph Attention Network (GAT) layer for CROWN's bipartite user-news graph.

    Args:
        units: Output dimension for both user and news embeddings
        num_heads: Number of attention heads
        dropout_rate: Dropout rate
        activation: Activation function
        seed: Random seed
        use_bias: Whether to use bias in linear transformations
        alpha: LeakyReLU negative slope for attention mechanism
        concat_heads: Whether to concatenate or average multi-head outputs
    """

    def __init__(
        self,
        units,
        num_heads=1,
        dropout_rate=0.0,
        activation="relu",
        seed=42,
        use_bias=True,
        alpha=0.2,
        concat_heads=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.activation = keras.activations.get(activation)
        self.seed = seed
        self.use_bias = use_bias
        self.alpha = alpha
        self.concat_heads = concat_heads

        if concat_heads:
            assert units % num_heads == 0, (
                "units must be divisible by num_heads when concat_heads=True"
            )
            self.head_dim = units // num_heads
        else:
            self.head_dim = units

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                "GraphAttentionLayer expects 3 inputs: (user_features, news_features, adjacency_matrix)"
            )

        user_features_shape, news_features_shape, _ = input_shape
        user_dim = user_features_shape[-1]
        news_dim = news_features_shape[-1]

        self.W_user = self.add_weight(
            name="W_user",
            shape=(self.num_heads, user_dim, self.head_dim),
            initializer=keras.initializers.GlorotUniform(seed=self.seed),
            trainable=True,
        )
        self.W_news = self.add_weight(
            name="W_news",
            shape=(self.num_heads, news_dim, self.head_dim),
            initializer=keras.initializers.GlorotUniform(seed=self.seed),
            trainable=True,
        )
        self.a_user_news = self.add_weight(
            name="a_user_news",
            shape=(self.num_heads, self.head_dim * 2, 1),
            initializer=keras.initializers.GlorotUniform(seed=self.seed),
            trainable=True,
        )
        self.a_news_user = self.add_weight(
            name="a_news_user",
            shape=(self.num_heads, self.head_dim * 2, 1),
            initializer=keras.initializers.GlorotUniform(seed=self.seed),
            trainable=True,
        )
        if self.use_bias:
            self.b_user = self.add_weight(
                name="b_user",
                shape=(self.units if self.concat_heads else self.head_dim,),
                initializer=keras.initializers.Zeros(),
                trainable=True,
            )
            self.b_news = self.add_weight(
                name="b_news",
                shape=(self.units if self.concat_heads else self.head_dim,),
                initializer=keras.initializers.Zeros(),
                trainable=True,
            )
        self.dropout = layers.Dropout(self.dropout_rate, seed=self.seed)
        super().build(input_shape)

    def compute_attention(
        self, queries, keys, attention_weights, adjacency_mask, training=None
    ):
        batch_size = ops.shape(queries)[0]
        num_queries = ops.shape(queries)[2]
        num_keys = ops.shape(keys)[2]

        queries_exp = ops.expand_dims(queries, axis=3)
        keys_exp = ops.expand_dims(keys, axis=2)

        queries_broadcast = ops.broadcast_to(
            queries_exp,
            (batch_size, self.num_heads, num_queries, num_keys, self.head_dim),
        )
        keys_broadcast = ops.broadcast_to(
            keys_exp, (batch_size, self.num_heads, num_queries, num_keys, self.head_dim)
        )

        concat_qk = ops.concatenate([queries_broadcast, keys_broadcast], axis=-1)
        attention_scores = ops.squeeze(
            ops.matmul(concat_qk, ops.expand_dims(attention_weights, axis=0)), axis=-1
        )
        attention_scores = ops.leaky_relu(attention_scores, negative_slope=self.alpha)

        adjacency_mask_exp = ops.expand_dims(adjacency_mask, axis=1)
        masked_scores = ops.where(adjacency_mask_exp > 0, attention_scores, -1e9)
        attention_coeffs = ops.softmax(masked_scores, axis=-1)
        attention_coeffs = self.dropout(attention_coeffs, training=training)
        return attention_coeffs

    def call(self, inputs, training=None):
        user_features, news_features, adjacency_matrix = inputs
        batch_size = ops.shape(user_features)[0]
        num_users = ops.shape(user_features)[1]
        num_news = ops.shape(news_features)[1]

        user_to_news = adjacency_matrix[:, :num_users, num_users:]
        news_to_user = adjacency_matrix[:, num_users:, :num_users]

        user_transformed = ops.einsum("bud,hdk->bhuk", user_features, self.W_user)
        news_transformed = ops.einsum("bnd,hdk->bhnk", news_features, self.W_news)

        user_news_attention = self.compute_attention(
            user_transformed,
            news_transformed,
            self.a_user_news,
            user_to_news,
            training=training,
        )
        user_updates = ops.einsum(
            "bhun,bhnk->bhuk", user_news_attention, news_transformed
        )

        news_user_attention = self.compute_attention(
            news_transformed,
            user_transformed,
            self.a_news_user,
            news_to_user,
            training=training,
        )
        news_updates = ops.einsum(
            "bhnu,bhuk->bhnk", news_user_attention, user_transformed
        )

        if self.concat_heads:
            updated_users = ops.reshape(
                user_updates, (batch_size, num_users, self.units)
            )
            updated_news = ops.reshape(news_updates, (batch_size, num_news, self.units))
        else:
            updated_users = ops.mean(user_updates, axis=1)
            updated_news = ops.mean(news_updates, axis=1)

        if self.use_bias:
            updated_users = updated_users + self.b_user
            updated_news = updated_news + self.b_news

        updated_users = self.activation(updated_users)
        updated_news = self.activation(updated_news)
        updated_users = self.dropout(updated_users, training=training)
        updated_news = self.dropout(updated_news, training=training)

        return updated_users, updated_news

    def compute_output_shape(self, input_shape):
        user_shape, news_shape, _ = input_shape
        output_dim = self.units if self.concat_heads else self.head_dim
        return (
            (user_shape[0], user_shape[1], output_dim),
            (news_shape[0], news_shape[1], output_dim),
        )
