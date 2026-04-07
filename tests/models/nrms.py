#!/usr/bin/env python3
"""Comprehensive test script for NRMS model implementation."""

import os
import sys
from pathlib import Path

# Set JAX as Keras backend before importing keras
os.environ["KERAS_BACKEND"] = "jax"

# Add the project root and src to Python path to enable imports
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))

import keras
import numpy as np

from src.frameworks.keras.models.nrms import NRMS


def create_dummy_data(batch_size=2, vocab_size=10000):
    """Create dummy data for testing."""
    data = {
        # Training format inputs
        "hist_tokens": np.random.randint(0, 1000, (batch_size, 50, 50), dtype=np.int32),
        "cand_tokens": np.random.randint(0, 1000, (batch_size, 5, 50), dtype=np.int32),
        # Single candidate format
        "history_tokens": np.random.randint(
            0, 1000, (batch_size, 50, 50), dtype=np.int32
        ),
        "single_candidate_tokens": np.random.randint(
            0, 1000, (batch_size, 50), dtype=np.int32
        ),
        # Labels for training
        "labels": np.zeros((batch_size, 5), dtype=np.float32),
    }
    # Set first item as positive for each batch
    data["labels"][:, 0] = 1.0
    return data


def test_nrms_components(nrms_model, batch_size=2):
    """Test individual NRMS components."""
    print("\n" + "=" * 50)
    print("TESTING INDIVIDUAL COMPONENTS")
    print("=" * 50)

    # Test news encoder
    print("\n1. Testing news encoder...")
    news_input = np.random.randint(0, 1000, (batch_size, 50), dtype=np.int32)
    news_output = nrms_model.news_encoder(news_input, training=False)
    print(f"   ✓ News encoder output shape: {news_output.shape}")
    assert news_output.shape == (
        batch_size,
        300,
    ), f"Expected shape (batch_size, 300), got {news_output.shape}"

    # Test user encoder
    print("\n2. Testing user encoder...")
    history_input = np.random.randint(0, 1000, (batch_size, 50, 50), dtype=np.int32)
    user_output = nrms_model.user_encoder(history_input, training=False)
    print(f"   ✓ User encoder output shape: {user_output.shape}")
    assert user_output.shape == (
        batch_size,
        300,
    ), f"Expected shape (batch_size, 300), got {user_output.shape}"

    # Test scorer component
    print("\n3. Testing scorer component...")

    # Test training batch scoring
    hist_tokens = np.random.randint(0, 1000, (batch_size, 50, 50), dtype=np.int32)
    cand_tokens = np.random.randint(0, 1000, (batch_size, 5, 50), dtype=np.int32)

    training_scores = nrms_model.scorer.score_training_batch(
        hist_tokens, cand_tokens, training=False
    )
    print(f"   ✓ Training batch scores shape: {training_scores.shape}")
    assert training_scores.shape == (
        batch_size,
        5,
    ), f"Expected shape (batch_size, 5), got {training_scores.shape}"

    # Verify logit output
    assert np.all(np.isfinite(training_scores)), "Training scores contain NaN or inf"
    print("   ✓ Training scores are finite (raw logits)")

    # Test single candidate scoring
    single_cand = np.random.randint(0, 1000, (batch_size, 50), dtype=np.int32)
    single_scores = nrms_model.scorer.score_single_candidate(
        hist_tokens, single_cand, training=False
    )
    print(f"   ✓ Single candidate scores shape: {single_scores.shape}")
    assert single_scores.shape == (
        batch_size,
        1,
    ), f"Expected shape (batch_size, 1), got {single_scores.shape}"

    # Test multiple candidates scoring
    multi_scores = nrms_model.scorer.score_multiple_candidates(
        hist_tokens, cand_tokens, training=False
    )
    print(f"   ✓ Multiple candidates scores shape: {multi_scores.shape}")
    assert multi_scores.shape == (
        batch_size,
        5,
    ), f"Expected shape (batch_size, 5), got {multi_scores.shape}"

    print("\n✅ All component tests passed!")


def test_nrms_call_method(nrms_model, batch_size=2):
    """Test the main call method with different input formats."""
    print("\n" + "=" * 50)
    print("TESTING MAIN CALL METHOD")
    print("=" * 50)

    dummy_data = create_dummy_data(batch_size)

    # Test 1: Training mode with training format
    print("\n1. Testing training mode...")
    training_inputs = {
        "hist_tokens": dummy_data["hist_tokens"],
        "cand_tokens": dummy_data["cand_tokens"],
    }

    training_output = nrms_model(training_inputs, training=True)
    print(f"   ✓ Training output shape: {training_output.shape}")
    assert training_output.shape == (
        batch_size,
        5,
    ), f"Expected shape (batch_size, 5), got {training_output.shape}"

    # Verify logit output
    assert np.all(np.isfinite(training_output)), "Training scores contain NaN or inf"
    print("   ✓ Training outputs are valid logits")

    # Test 2: Inference mode with single candidate
    print("\n2. Testing inference mode with single candidate...")
    single_cand_inputs = {
        "history_tokens": dummy_data["history_tokens"],
        "single_candidate_tokens": dummy_data["single_candidate_tokens"],
    }

    single_output = nrms_model(single_cand_inputs, training=False)
    print(f"   ✓ Single candidate output shape: {single_output.shape}")
    assert single_output.shape == (
        batch_size,
        1,
    ), f"Expected shape (batch_size, 1), got {single_output.shape}"

    # Verify sigmoid output (should be between 0 and 1)
    assert np.all(single_output >= 0) and np.all(single_output <= 1), (
        "Sigmoid outputs should be in [0, 1]"
    )
    print("   ✓ Single candidate outputs are valid sigmoid scores")

    # Test 3: Inference mode with multiple candidates (validation format)
    print("\n3. Testing inference mode with multiple candidates...")
    multi_cand_inputs = {
        "hist_tokens": dummy_data["hist_tokens"],
        "cand_tokens": dummy_data["cand_tokens"],
    }

    multi_output = nrms_model(multi_cand_inputs, training=False)
    print(f"   ✓ Multiple candidates output shape: {multi_output.shape}")
    assert multi_output.shape == (
        batch_size,
        5,
    ), f"Expected shape (batch_size, 5), got {multi_output.shape}"

    # Verify sigmoid outputs (should be between 0 and 1)
    assert np.all(multi_output >= 0) and np.all(multi_output <= 1), (
        "Sigmoid outputs should be in [0, 1]"
    )
    print("   ✓ Multiple candidate outputs are valid sigmoid scores")

    print("\n✅ All call method tests passed!")


def test_nrms_compilation(nrms_model, batch_size=2):
    """Test model compilation and training step."""
    print("\n" + "=" * 50)
    print("TESTING MODEL COMPILATION AND TRAINING")
    print("=" * 50)

    # Compile the model
    print("\n1. Compiling model...")
    nrms_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.CategoricalAccuracy()],
    )
    print("   ✓ Model compiled successfully")

    # Test a training step
    print("\n2. Testing training step...")
    dummy_data = create_dummy_data(batch_size)

    # Prepare inputs and labels
    x = {
        "hist_tokens": dummy_data["hist_tokens"],
        "cand_tokens": dummy_data["cand_tokens"],
    }
    y = dummy_data["labels"]

    # Perform one training step
    loss = nrms_model.train_on_batch(x, y)
    print(f"   ✓ Training step completed, loss: {loss}")

    # Test evaluation
    print("\n3. Testing evaluation...")
    eval_results = nrms_model.evaluate(x, y, verbose=0)
    print(f"   ✓ Evaluation completed, metrics: {eval_results}")

    print("\n✅ Compilation and training tests passed!")


def test_nrms_model():
    """Comprehensive test of NRMS model."""

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
        "embedding_size": 300,
        "multiheads": 16,
        "head_dim": 16,
        "attention_hidden_dim": 200,
        "dropout_rate": 0.2,
        "seed": 42,
        "max_title_length": 50,
        "max_history_length": 50,
        "max_impressions_length": 5,
        "process_user_id": False,
    }

    try:
        # Create NRMS model
        print("=" * 50)
        print("CREATING NRMS MODEL")
        print("=" * 50)
        nrms_model = NRMS(**model_params)
        print("✓ NRMS model created successfully")

        # Check that model is built
        assert nrms_model.news_encoder is not None, "News encoder not initialized"
        assert nrms_model.user_encoder is not None, "User encoder not initialized"
        assert nrms_model.scorer is not None, "Scorer not initialized"
        assert nrms_model.training_model is not None, "Training model not initialized"
        assert nrms_model.scorer_model is not None, "Scorer model not initialized"
        print("✓ All model components initialized")

        # Run component tests
        test_nrms_components(nrms_model)

        # Run call method tests
        test_nrms_call_method(nrms_model)

        # Run compilation and training tests
        test_nrms_compilation(nrms_model)

        # Print model summary
        print("\n" + "=" * 50)
        print("MODEL SUMMARY")
        print("=" * 50)
        print("\nMain NRMS Model:")
        nrms_model.summary()

        print("\n🎉 All NRMS tests passed successfully!")

    except Exception as e:
        print(f"\n❌ Error testing NRMS model: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    keras.utils.set_random_seed(42)

    print("\n" + "=" * 60)
    print("NRMS MODEL COMPREHENSIVE TEST SUITE")
    print(f"Backend: {keras.backend.backend()}")
    print("=" * 60)

    success = test_nrms_model()

    if success:
        print("\n" + "=" * 60)
        print("✅ NRMS MODEL IS FULLY FUNCTIONAL AND READY FOR TRAINING!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ NRMS MODEL IMPLEMENTATION NEEDS FIXES")
        print("=" * 60)
