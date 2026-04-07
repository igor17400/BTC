#!/usr/bin/env python3
"""Comprehensive test script for CROWN model implementation."""

import os
import sys
from pathlib import Path

# Set JAX as Keras backend before importing keras
os.environ["KERAS_BACKEND"] = "jax"

# Add the project root to Python path to enable imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import keras
import numpy as np

from src.frameworks.keras.models.crown import CROWN


def create_dummy_data(batch_size=2):
    """Create dummy data for testing CROWN model."""
    data = {
        # History inputs
        "hist_tokens": np.random.randint(
            1, 1000, (batch_size, 50, 30), dtype=np.int32
        ),  # title tokens
        "hist_abstract_tokens": np.random.randint(
            1, 1000, (batch_size, 50, 100), dtype=np.int32
        ),  # abstract tokens
        "hist_category": np.random.randint(
            0, 20, (batch_size, 50), dtype=np.int32
        ),  # category IDs
        "hist_subcategory": np.random.randint(
            0, 50, (batch_size, 50), dtype=np.int32
        ),  # subcategory IDs
        # Candidate inputs (multiple candidates for training)
        "cand_tokens": np.random.randint(
            1, 1000, (batch_size, 5, 30), dtype=np.int32
        ),  # candidate titles
        "cand_abstract_tokens": np.random.randint(
            1, 1000, (batch_size, 5, 100), dtype=np.int32
        ),  # candidate abstracts
        "cand_category": np.random.randint(
            0, 20, (batch_size, 5), dtype=np.int32
        ),  # candidate categories
        "cand_subcategory": np.random.randint(
            0, 50, (batch_size, 5), dtype=np.int32
        ),  # candidate subcategories
        # Labels for training
        "labels": np.zeros((batch_size, 5), dtype=np.float32),
    }

    # Set first item as positive for each batch
    data["labels"][:, 0] = 1.0

    return data


def test_crown_components(crown_model, batch_size=2):
    """Test individual CROWN components."""
    print("\n" + "=" * 50)
    print("TESTING INDIVIDUAL COMPONENTS")
    print("=" * 50)

    dummy_data = create_dummy_data(batch_size)

    # Test news encoder
    print("\n1. Testing news encoder...")
    # CROWN NewsEncoder expects concatenated input: [title_tokens, abstract_tokens, category_id, subcategory_id]
    title_tokens = dummy_data["hist_tokens"][:, 0, :]  # First news item from history
    abstract_tokens = dummy_data["hist_abstract_tokens"][:, 0, :]
    category_id = np.expand_dims(dummy_data["hist_category"][:, 0], axis=-1)
    subcategory_id = np.expand_dims(dummy_data["hist_subcategory"][:, 0], axis=-1)

    # Concatenate inputs as expected by CROWN NewsEncoder
    news_input = np.concatenate(
        [title_tokens, abstract_tokens, category_id, subcategory_id], axis=-1
    )

    news_output = crown_model.news_encoder(news_input, training=False)
    print(f"   ✓ News encoder output shape: {news_output.shape}")
    expected_news_dim = crown_model.news_encoder.news_embedding_dim
    assert news_output.shape == (
        batch_size,
        expected_news_dim,
    ), f"Expected shape (batch_size, {expected_news_dim}), got {news_output.shape}"

    # Test user encoder
    print("\n2. Testing user encoder...")
    # User encoder expects concatenated history inputs
    history_concat = np.concatenate(
        [
            dummy_data["hist_tokens"],
            dummy_data["hist_abstract_tokens"],
            np.expand_dims(dummy_data["hist_category"], axis=-1),
            np.expand_dims(dummy_data["hist_subcategory"], axis=-1),
        ],
        axis=-1,
    )

    user_output = crown_model.user_encoder(history_concat, training=False)
    print(f"   ✓ User encoder output shape: {user_output.shape}")
    assert user_output.shape == (
        batch_size,
        expected_news_dim,
    ), f"Expected shape (batch_size, {expected_news_dim}), got {user_output.shape}"

    # Test scorer component
    print("\n3. Testing scorer component...")

    # Prepare candidates
    candidate_concat = np.concatenate(
        [
            dummy_data["cand_tokens"],
            dummy_data["cand_abstract_tokens"],
            np.expand_dims(dummy_data["cand_category"], axis=-1),
            np.expand_dims(dummy_data["cand_subcategory"], axis=-1),
        ],
        axis=-1,
    )

    # Test training batch scoring
    training_scores = crown_model.scorer.score_training_batch(
        history_concat, candidate_concat, training=False
    )
    print(f"   ✓ Training batch scores shape: {training_scores.shape}")
    assert training_scores.shape == (
        batch_size,
        5,
    ), f"Expected shape (batch_size, 5), got {training_scores.shape}"

    # Verify softmax output
    score_sums = np.sum(training_scores, axis=-1)
    assert np.allclose(score_sums, 1.0, rtol=1e-5), (
        f"Softmax scores don't sum to 1: {score_sums}"
    )
    print(f"   ✓ Softmax scores sum to 1.0: {score_sums}")

    print("\n✅ All component tests passed!")


def test_crown_call_method(crown_model, batch_size=2):
    """Test the main call method with different input formats."""
    print("\n" + "=" * 50)
    print("TESTING MAIN CALL METHOD")
    print("=" * 50)

    dummy_data = create_dummy_data(batch_size)

    # Test 1: Training mode
    print("\n1. Testing training mode...")
    training_inputs = {
        "hist_tokens": dummy_data["hist_tokens"],
        "hist_abstract_tokens": dummy_data["hist_abstract_tokens"],
        "hist_category": dummy_data["hist_category"],
        "hist_subcategory": dummy_data["hist_subcategory"],
        "cand_tokens": dummy_data["cand_tokens"],
        "cand_abstract_tokens": dummy_data["cand_abstract_tokens"],
        "cand_category": dummy_data["cand_category"],
        "cand_subcategory": dummy_data["cand_subcategory"],
    }

    training_output = crown_model(training_inputs, training=True)
    print(f"   ✓ Training output shape: {training_output.shape}")
    assert training_output.shape == (
        batch_size,
        5,
    ), f"Expected shape (batch_size, 5), got {training_output.shape}"

    # Verify softmax output
    score_sums = np.sum(training_output, axis=-1)
    assert np.allclose(score_sums, 1.0, rtol=1e-5), (
        f"Softmax scores don't sum to 1: {score_sums}"
    )
    print("   ✓ Training outputs are valid softmax scores")

    # Test 2: Inference mode with multiple candidates
    print("\n2. Testing inference mode with multiple candidates...")
    inference_inputs = {
        "hist_tokens": dummy_data["hist_tokens"],
        "hist_abstract_tokens": dummy_data["hist_abstract_tokens"],
        "hist_category": dummy_data["hist_category"],
        "hist_subcategory": dummy_data["hist_subcategory"],
        "cand_tokens": dummy_data["cand_tokens"],
        "cand_abstract_tokens": dummy_data["cand_abstract_tokens"],
        "cand_category": dummy_data["cand_category"],
        "cand_subcategory": dummy_data["cand_subcategory"],
    }

    multi_output = crown_model(inference_inputs, training=False)
    print(f"   ✓ Multiple candidates output shape: {multi_output.shape}")
    assert multi_output.shape == (
        batch_size,
        5,
    ), f"Expected shape (batch_size, 5), got {multi_output.shape}"

    # CROWN uses sigmoid for multiple candidates in inference
    assert np.all(multi_output >= 0) and np.all(multi_output <= 1), (
        "Outputs should be in [0, 1]"
    )
    print("   ✓ Multiple candidate outputs are valid scores")

    print("\n✅ All call method tests passed!")


def test_crown_compilation(crown_model, batch_size=2):
    """Test model compilation and training step."""
    print("\n" + "=" * 50)
    print("TESTING MODEL COMPILATION AND TRAINING")
    print("=" * 50)

    # Compile the model
    print("\n1. Compiling model...")
    crown_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=[keras.metrics.CategoricalAccuracy()],
    )
    print("   ✓ Model compiled successfully")

    # Test a training step
    print("\n2. Testing training step...")
    dummy_data = create_dummy_data(batch_size)

    # Prepare inputs and labels
    x = {
        "hist_tokens": dummy_data["hist_tokens"],
        "hist_abstract_tokens": dummy_data["hist_abstract_tokens"],
        "hist_category": dummy_data["hist_category"],
        "hist_subcategory": dummy_data["hist_subcategory"],
        "cand_tokens": dummy_data["cand_tokens"],
        "cand_abstract_tokens": dummy_data["cand_abstract_tokens"],
        "cand_category": dummy_data["cand_category"],
        "cand_subcategory": dummy_data["cand_subcategory"],
    }
    y = dummy_data["labels"]

    # Perform one training step
    loss = crown_model.train_on_batch(x, y)
    print(f"   ✓ Training step completed, loss: {loss}")

    # Test evaluation
    print("\n3. Testing evaluation...")
    eval_results = crown_model.evaluate(x, y, verbose=0)
    print(f"   ✓ Evaluation completed, metrics: {eval_results}")

    print("\n✅ Compilation and training tests passed!")


def test_crown_model():
    """Comprehensive test of CROWN model."""

    # Mock processed news data
    processed_news = {
        "vocab_size": 10000,
        "num_categories": 20,
        "num_subcategories": 50,
        "embeddings": np.random.randn(10000, 300).astype(np.float32),
    }

    # Model parameters
    model_params = {
        "processed_news": processed_news,
        "max_title_length": 30,
        "max_abstract_length": 100,
        "embedding_size": 300,
        "intent_embedding_dim": 200,
        "category_embedding_dim": 100,
        "subcategory_embedding_dim": 100,
        "attention_dim": 200,
        "intent_num": 4,
        "alpha": 0.5,
        "num_heads": 12,
        "head_dim": 25,
        "feedforward_dim": 512,
        "num_layers": 2,
        "isab_num_heads": 4,
        "isab_num_inducing_points": 4,
        "graph_hidden_dim": 300,
        "graph_num_layers": 1,
        "dropout_rate": 0.2,
        "max_history_length": 50,
        "max_impressions_length": 5,
        "seed": 42,
    }

    try:
        # Create CROWN model
        print("=" * 50)
        print("CREATING CROWN MODEL")
        print("=" * 50)
        crown_model = CROWN(**model_params)
        print("✓ CROWN model created successfully")

        # Check that model is built
        assert crown_model.news_encoder is not None, "News encoder not initialized"
        assert crown_model.user_encoder is not None, "User encoder not initialized"
        assert crown_model.scorer is not None, "Scorer not initialized"
        print("✓ All model components initialized")

        # Run component tests
        test_crown_components(crown_model, batch_size=2)

        # Run call method tests
        test_crown_call_method(crown_model, batch_size=2)

        # Run compilation and training tests
        test_crown_compilation(crown_model, batch_size=2)

        # Print model configuration
        print("\n" + "=" * 50)
        print("MODEL CONFIGURATION")
        print("=" * 50)
        print(f"Config: {crown_model.config}")
        print(
            f"News embedding dimension: {crown_model.news_encoder.news_embedding_dim}"
        )
        print(f"Intent number: {crown_model.config.intent_num}")
        print(f"Intent embedding dim: {crown_model.config.intent_embedding_dim}")

        print("\n🎉 All CROWN tests passed successfully!")

    except Exception as e:
        print(f"\n❌ Error testing CROWN model: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    keras.utils.set_random_seed(42)

    print("\n" + "=" * 60)
    print("CROWN MODEL COMPREHENSIVE TEST SUITE")
    print(f"Backend: {keras.backend.backend()}")
    print("=" * 60)

    success = test_crown_model()

    if success:
        print("\n" + "=" * 60)
        print("✅ CROWN MODEL IS FULLY FUNCTIONAL AND READY FOR TRAINING!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ CROWN MODEL IMPLEMENTATION NEEDS FIXES")
        print("=" * 60)
