import keras
from keras import layers, ops

# ---------------------------------------------------------------------------
# Initializers
# ---------------------------------------------------------------------------


class GlorotUniformMHA(keras.initializers.Initializer):
    """GlorotUniform with the *correct* fan computation for MHA EinsumDense kernels.

    Keras 3's ``MultiHeadAttention`` stores Q/K/V projections as 3D einsum
    kernels of shape ``(input_dim, num_heads, head_dim)`` and the output
    projection as ``(num_heads, head_dim, output_dim)``. The default
    ``GlorotUniform`` treats these as convolution kernels and computes
    ``fan_in = input_dim * num_heads`` and ``fan_out = input_dim * head_dim``,
    which is **wrong** — it gives an init scale roughly 4× smaller than what
    JAX/Flax and PyTorch use. The result is severely underscaled MHA outputs
    and slow training.

    This initializer uses the correct fans:
        For ``(input_dim, num_heads, head_dim)``: fan_in=input_dim,
            fan_out=num_heads*head_dim.
        For ``(num_heads, head_dim, output_dim)``: fan_in=num_heads*head_dim,
            fan_out=output_dim.
        For 2D kernels: standard ``(fan_in, fan_out)``.
    """

    def __call__(self, shape, dtype=None):
        if len(shape) == 3:
            # Q/K/V kernel: (input_dim, num_heads, head_dim) — first dim is large
            # Output kernel: (num_heads, head_dim, output_dim) — last dim is large
            first = int(shape[0])
            mid_last = int(shape[1]) * int(shape[2])
            if first >= mid_last:
                # Q/K/V style: fan_in = input_dim, fan_out = num_heads * head_dim
                fan_in, fan_out = first, mid_last
            else:
                # Output proj style: fan_in = num_heads * head_dim, fan_out = output_dim
                fan_in, fan_out = int(shape[0]) * int(shape[1]), int(shape[2])
        elif len(shape) == 2:
            fan_in, fan_out = int(shape[0]), int(shape[1])
        else:
            fan_in = int(shape[0])
            fan_out = int(shape[-1])
        limit = float((6.0 / (fan_in + fan_out)) ** 0.5)
        return keras.random.uniform(shape, minval=-limit, maxval=limit, dtype=dtype)

    def get_config(self):
        return {}


# ---------------------------------------------------------------------------
# Pure-function masking utilities (isomorphic with JAX/PyTorch)
# ---------------------------------------------------------------------------


def compute_mask(inputs):
    """Compute a boolean mask where non-zero positions are True."""
    return ops.cast(ops.not_equal(inputs, 0), "float32")


def overwrite_mask(values, mask):
    """Zero out positions indicated by a mask."""
    return values * ops.expand_dims(mask, axis=-1)


class AdditiveAttention(layers.Layer):
    """
    Soft-alignment-based attention layer.

    This layer computes a weighted sum of the input sequence, where the weights are
    learned during training. It's a common attention mechanism used in various
    NLP and recommendation models.

    Args:
        dim (int): The hidden dimension of the attention mechanism.
        seed (int): Random seed for reproducibility of initializers.
    """

    def __init__(self, query_vec_dim=200, seed=0, **kwargs):
        super().__init__(**kwargs)
        self.dim = query_vec_dim
        self.seed = seed
        self.supports_masking = True

    def build(self, input_shape):
        """
        Create the weights for the layer.

        This method creates the three trainable variables: W, b, and q, which are
        used to compute the attention scores.
        """
        if not isinstance(input_shape, tuple) or len(input_shape) != 3:
            raise ValueError("A `AttLayer2` layer should be called on a 3D tensor.")

        # Weight matrix for the dense transformation
        self.W = self.add_weight(
            name="W",
            shape=(input_shape[-1], self.dim),
            initializer=keras.initializers.GlorotUniform(seed=self.seed),
            trainable=True,
        )
        # Bias vector for the dense transformation
        self.b = self.add_weight(
            name="b",
            shape=(self.dim,),
            initializer=keras.initializers.Zeros(),
            trainable=True,
        )
        # Query vector to compute attention scores
        self.q = self.add_weight(
            name="q",
            shape=(self.dim, 1),
            initializer=keras.initializers.GlorotUniform(seed=self.seed),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, mask=None):
        """
        Defines the forward pass of the layer.

        Args:
            inputs (keras.KerasTensor): The input 3D tensor `(batch_size, time_steps, features)`.
            mask (keras.KerasTensor, optional): A boolean mask `(batch_size, time_steps)`
                                                to ignore certain time steps.

        Returns:
            keras.KerasTensor: A 2D tensor `(batch_size, features)` representing the
                weighted sum of the input sequence.
        """
        # 1. Dense transformation and non-linearity
        attention_hidden = ops.tanh(ops.matmul(inputs, self.W) + self.b)

        # 2. Compute attention scores
        attention_scores = ops.matmul(attention_hidden, self.q)
        attention_scores = ops.squeeze(attention_scores, axis=-1)

        # 3. Apply exp and mask (following original paper implementation)
        if mask is None:
            attention = ops.exp(attention_scores)
        else:
            attention = ops.exp(attention_scores) * ops.cast(
                mask, dtype=self.compute_dtype
            )

        # 4. Normalize attention weights
        attention_weights = attention / (
            ops.sum(attention, axis=-1, keepdims=True) + keras.backend.epsilon()
        )

        # 5. Compute weighted sum of inputs
        attention_weights_expanded = ops.expand_dims(attention_weights, axis=-1)
        weighted_input = inputs * attention_weights_expanded

        return ops.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        """Computes the output shape of the layer."""
        return input_shape[0], input_shape[-1]


class AttentivePoolingQKY(layers.Layer):
    """Attentive pooling where attention keys differ from values.

    Computes attention weights from ``key_input`` (e.g. concatenated
    content + popularity embeddings) but returns a weighted sum of
    ``value_input`` (e.g. content embeddings only).

    Used in PP-Rec's Content-Popularity Joint Attention (CPJA).

    Args:
        query_vec_dim: Hidden dimension for the attention MLP.
        seed: Random seed for reproducibility.
    """

    def __init__(self, query_vec_dim=200, seed=0, **kwargs):
        super().__init__(**kwargs)
        self.dim = query_vec_dim
        self.seed = seed

    def build(self, input_shape):
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
            raise ValueError(
                "AttentivePoolingQKY expects two inputs: [key_input, value_input]."
            )
        key_shape = input_shape[0]

        self.W = self.add_weight(
            name="W",
            shape=(key_shape[-1], self.dim),
            initializer=keras.initializers.GlorotUniform(seed=self.seed),
            trainable=True,
        )
        self.b = self.add_weight(
            name="b",
            shape=(self.dim,),
            initializer=keras.initializers.Zeros(),
            trainable=True,
        )
        self.q = self.add_weight(
            name="q",
            shape=(self.dim, 1),
            initializer=keras.initializers.GlorotUniform(seed=self.seed),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, mask=None):
        """Forward pass.

        Args:
            inputs: List of [key_input, value_input].
                key_input: (batch, seq_len, key_dim) — used for attention weights.
                value_input: (batch, seq_len, val_dim) — aggregated by attention.
            mask: Optional boolean mask (batch, seq_len).

        Returns:
            Weighted sum of value_input: (batch, val_dim).
        """
        key_input, value_input = inputs

        attention_hidden = ops.tanh(ops.matmul(key_input, self.W) + self.b)
        attention_scores = ops.squeeze(ops.matmul(attention_hidden, self.q), axis=-1)

        if mask is None:
            attention = ops.exp(attention_scores)
        else:
            attention = ops.exp(attention_scores) * ops.cast(
                mask, dtype=self.compute_dtype
            )

        attention_weights = attention / (
            ops.sum(attention, axis=-1, keepdims=True) + keras.backend.epsilon()
        )

        attention_weights_expanded = ops.expand_dims(attention_weights, axis=-1)
        weighted_input = value_input * attention_weights_expanded
        return ops.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        _, value_shape = input_shape
        return (value_shape[0], value_shape[-1])


class ComputeMasking(layers.Layer):
    """Compute if inputs contains zero value.

    Returns:
        bool tensor: True for values not equal to zero.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """Call method for ComputeMasking.

        Args:
            inputs (object): input tensor.

        Returns:
            bool tensor: True for values not equal to zero.
        """
        mask = ops.not_equal(inputs, 0)
        return ops.cast(mask, self.compute_dtype)

    def compute_output_shape(self, input_shape):
        return input_shape


class OverwriteMasking(layers.Layer):
    """Set values at specific positions to zero.

    Args:
        inputs (list): value tensor and mask tensor.

    Returns:
        object: tensor after setting values to zero.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        """Call method for OverwriteMasking.

        Args:
            inputs (list): value tensor and mask tensor.

        Returns:
            object: tensor after setting values to zero.
        """
        return inputs[0] * ops.expand_dims(inputs[1], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[0]


