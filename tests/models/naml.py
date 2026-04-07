#!/usr/bin/env python3
"""Comprehensive test script for NAML model implementation."""

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

from src.frameworks.keras.models.naml import NAML


def create_dummy_data(batch_size=2, vocab_size=10000):
    """Create dummy data for testing NAML."""
    # NAML expects concatenated inputs: [title, abstract, category, subcategory]
    # Total length = max_title_length + max_abstract_length + 2
    total_length = 30 + 100 + 2  # 132

    data = {
        # Training format inputs (concatenated)
        "hist_concat": np.random.randint(
            0, 1000, (batch_size, 50, total_length), dtype=np.int32
        ),
        "cand_concat": np.random.randint(
            0, 1000, (batch_size, 5, total_length), dtype=np.int32
        ),
        # Set category and subcategory IDs (last 2 positions)
        # These should be smaller values for category/subcategory embeddings
        "hist_concat_with_cats": np.random.randint(
            0, 1000, (batch_size, 50, total_length), dtype=np.int32
        ),
        "cand_concat_with_cats": np.random.randint(
            0, 1000, (batch_size, 5, total_length), dtype=np.int32
        ),
        # Single candidate format
        "history_concat": np.random.randint(
            0, 1000, (batch_size, 50, total_length), dtype=np.int32
        ),
        "single_candidate_concat": np.random.randint(
            0, 1000, (batch_size, total_length), dtype=np.int32
        ),
        # Labels for training
        "labels": np.zeros((batch_size, 5), dtype=np.float32),
    }

    # Set proper category and subcategory values
    for key in ["hist_concat_with_cats", "cand_concat_with_cats", "history_concat"]:
        if "hist" in key or "history" in key:
            # For history: shape is (batch_size, history_length, total_length)
            data[key][:, :, -2] = np.random.randint(
                0, 20, data[key][:, :, -2].shape
            )  # category
            data[key][:, :, -1] = np.random.randint(
                0, 50, data[key][:, :, -1].shape
            )  # subcategory
        else:
            # For candidates: shape is (batch_size, num_candidates, total_length) or (batch_size, total_length)
            if len(data[key].shape) == 3:
                data[key][:, :, -2] = np.random.randint(
                    0, 20, data[key][:, :, -2].shape
                )
                data[key][:, :, -1] = np.random.randint(
                    0, 50, data[key][:, :, -1].shape
                )

    data["single_candidate_concat"][:, -2] = np.random.randint(
        0, 20, (batch_size,)
    )  # category
    data["single_candidate_concat"][:, -1] = np.random.randint(
        0, 50, (batch_size,)
    )  # subcategory

    # Set first item as positive for each batch
    data["labels"][:, 0] = 1.0

    return data


def test_naml_components(naml_model, batch_size=2):
    """Test individual NAML components."""
    print("\n" + "=" * 50)
    print("TESTING INDIVIDUAL COMPONENTS")
    print("=" * 50)

    # Test individual view encoders
    print("\n1. Testing view encoders...")

    # Title encoder
    title_input = np.random.randint(0, 1000, (batch_size, 30), dtype=np.int32)
    title_output = naml_model.title_encoder(title_input, training=False)
    print(f"   ✓ Title encoder output shape: {title_output.shape}")
    assert title_output.shape == (
        batch_size,
        400,
    ), f"Expected shape (batch_size, 400), got {title_output.shape}"

    # Abstract encoder
    abstract_input = np.random.randint(0, 1000, (batch_size, 100), dtype=np.int32)
    abstract_output = naml_model.abstract_encoder(abstract_input, training=False)
    print(f"   ✓ Abstract encoder output shape: {abstract_output.shape}")
    assert abstract_output.shape == (
        batch_size,
        400,
    ), f"Expected shape (batch_size, 400), got {abstract_output.shape}"

    # Category encoder
    category_input = np.random.randint(0, 20, (batch_size, 1), dtype=np.int32)
    category_output = naml_model.category_encoder(category_input, training=False)
    print(f"   ✓ Category encoder output shape: {category_output.shape}")
    assert category_output.shape == (
        batch_size,
        400,
    ), f"Expected shape (batch_size, 400), got {category_output.shape}"

    # Subcategory encoder
    subcategory_input = np.random.randint(0, 50, (batch_size, 1), dtype=np.int32)
    subcategory_output = naml_model.subcategory_encoder(
        subcategory_input, training=False
    )
    print(f"   ✓ Subcategory encoder output shape: {subcategory_output.shape}")
    assert subcategory_output.shape == (
        batch_size,
        400,
    ), f"Expected shape (batch_size, 400), got {subcategory_output.shape}"

    # Test news encoder (combines all views)
    print("\n2. Testing news encoder...")
    news_input = np.random.randint(0, 1000, (batch_size, 132), dtype=np.int32)
    news_input[:, -2] = np.random.randint(0, 20, (batch_size,))  # category
    news_input[:, -1] = np.random.randint(0, 50, (batch_size,))  # subcategory

    news_output = naml_model.news_encoder(news_input, training=False)
    print(f"   ✓ News encoder output shape: {news_output.shape}")
    assert news_output.shape == (
        batch_size,
        400,
    ), f"Expected shape (batch_size, 400), got {news_output.shape}"

    # Test user encoder
    print("\n3. Testing user encoder...")
    history_input = np.random.randint(0, 1000, (batch_size, 50, 132), dtype=np.int32)
    history_input[:, :, -2] = np.random.randint(0, 20, history_input[:, :, -2].shape)
    history_input[:, :, -1] = np.random.randint(0, 50, history_input[:, :, -1].shape)

    user_output = naml_model.user_encoder(history_input, training=False)
    print(f"   ✓ User encoder output shape: {user_output.shape}")
    assert user_output.shape == (
        batch_size,
        400,
    ), f"Expected shape (batch_size, 400), got {user_output.shape}"

    # Test scorer component
    print("\n4. Testing scorer component...")

    # Prepare inputs
    hist_concat = np.random.randint(0, 1000, (batch_size, 50, 132), dtype=np.int32)
    hist_concat[:, :, -2] = np.random.randint(0, 20, hist_concat[:, :, -2].shape)
    hist_concat[:, :, -1] = np.random.randint(0, 50, hist_concat[:, :, -1].shape)

    cand_concat = np.random.randint(0, 1000, (batch_size, 5, 132), dtype=np.int32)
    cand_concat[:, :, -2] = np.random.randint(0, 20, cand_concat[:, :, -2].shape)
    cand_concat[:, :, -1] = np.random.randint(0, 50, cand_concat[:, :, -1].shape)

    # Test training batch scoring
    training_scores = naml_model.scorer.score_training_batch(
        hist_concat, cand_concat, training=False
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
    single_cand = np.random.randint(0, 1000, (batch_size, 132), dtype=np.int32)
    single_cand[:, -2] = np.random.randint(0, 20, (batch_size,))
    single_cand[:, -1] = np.random.randint(0, 50, (batch_size,))

    single_scores = naml_model.scorer.score_single_candidate(
        hist_concat, single_cand, training=False
    )
    print(f"   ✓ Single candidate scores shape: {single_scores.shape}")
    assert single_scores.shape == (
        batch_size,
        1,
    ), f"Expected shape (batch_size, 1), got {single_scores.shape}"

    print("\n✅ All component tests passed!")


def test_naml_call_method(naml_model, batch_size=2):
    """Test the main call method with different input formats."""
    print("\n" + "=" * 50)
    print("TESTING MAIN CALL METHOD")
    print("=" * 50)

    # For NAML, we need to provide separate inputs for each component
    # Test 1: Training mode with separate inputs
    print("\n1. Testing training mode...")

    # Create separate inputs for training
    training_inputs = {
        "hist_tokens": np.random.randint(
            0, 1000, (batch_size, 50, 30), dtype=np.int32
        ),  # title tokens
        "hist_abstract_tokens": np.random.randint(
            0, 1000, (batch_size, 50, 100), dtype=np.int32
        ),  # abstract tokens
        "hist_category": np.random.randint(
            0, 20, (batch_size, 50), dtype=np.int32
        ),  # category IDs
        "hist_subcategory": np.random.randint(
            0, 50, (batch_size, 50), dtype=np.int32
        ),  # subcategory IDs
        "cand_tokens": np.random.randint(
            0, 1000, (batch_size, 5, 30), dtype=np.int32
        ),  # candidate titles
        "cand_abstract_tokens": np.random.randint(
            0, 1000, (batch_size, 5, 100), dtype=np.int32
        ),  # candidate abstracts
        "cand_category": np.random.randint(
            0, 20, (batch_size, 5), dtype=np.int32
        ),  # candidate categories
        "cand_subcategory": np.random.randint(
            0, 50, (batch_size, 5), dtype=np.int32
        ),  # candidate subcategories
    }

    training_output = naml_model(training_inputs, training=True)
    print(f"   ✓ Training output shape: {training_output.shape}")
    assert training_output.shape == (
        batch_size,
        5,
    ), f"Expected shape (batch_size, 5), got {training_output.shape}"

    # Verify logit output
    assert np.all(np.isfinite(training_output)), "Training scores contain NaN or inf"
    print("   ✓ Training outputs are valid logits")

    # Test 2: Using the compatibility training_model with concatenated inputs
    print("\n2. Testing compatibility training_model...")
    dummy_data = create_dummy_data(batch_size)

    # Use the training_model which expects concatenated inputs
    concat_hist = dummy_data["hist_concat_with_cats"]
    concat_cand = dummy_data["cand_concat_with_cats"]

    training_output = naml_model.training_model([concat_hist, concat_cand])
    print(f"   ✓ Training model output shape: {training_output.shape}")
    assert training_output.shape == (
        batch_size,
        5,
    ), f"Expected shape (batch_size, 5), got {training_output.shape}"

    # Verify logit output
    assert np.all(np.isfinite(training_output)), "Training scores contain NaN or inf"
    print("   ✓ Training model outputs are valid logits")

    # Test 3: Using the scorer_model for single candidate
    print("\n3. Testing scorer_model for single candidate...")

    concat_single_cand = dummy_data["single_candidate_concat"]

    scorer_output = naml_model.scorer_model([concat_hist[:, :, :], concat_single_cand])
    print(f"   ✓ Scorer model output shape: {scorer_output.shape}")
    assert scorer_output.shape == (
        batch_size,
        1,
    ), f"Expected shape (batch_size, 1), got {scorer_output.shape}"

    # Verify sigmoid output
    assert np.all(scorer_output >= 0) and np.all(scorer_output <= 1), (
        "Sigmoid outputs should be in [0, 1]"
    )
    print("   ✓ Scorer model outputs are valid sigmoid scores")

    # Test 4: Inference mode with multiple candidates using separate inputs
    print("\n4. Testing inference mode with multiple candidates...")

    # Create separate inputs for inference
    inference_inputs = {
        "hist_tokens": np.random.randint(0, 1000, (batch_size, 50, 30), dtype=np.int32),
        "hist_abstract_tokens": np.random.randint(
            0, 1000, (batch_size, 50, 100), dtype=np.int32
        ),
        "hist_category": np.random.randint(0, 20, (batch_size, 50), dtype=np.int32),
        "hist_subcategory": np.random.randint(0, 50, (batch_size, 50), dtype=np.int32),
        "cand_tokens": np.random.randint(0, 1000, (batch_size, 5, 30), dtype=np.int32),
        "cand_abstract_tokens": np.random.randint(
            0, 1000, (batch_size, 5, 100), dtype=np.int32
        ),
        "cand_category": np.random.randint(0, 20, (batch_size, 5), dtype=np.int32),
        "cand_subcategory": np.random.randint(0, 50, (batch_size, 5), dtype=np.int32),
    }

    multi_output = naml_model(inference_inputs, training=False)
    print(f"   ✓ Multiple candidates output shape: {multi_output.shape}")
    assert multi_output.shape == (
        batch_size,
        5,
    ), f"Expected shape (batch_size, 5), got {multi_output.shape}"

    # NAML uses sigmoid for multiple candidates in inference
    assert np.all(multi_output >= 0) and np.all(multi_output <= 1), (
        "Outputs should be in [0, 1]"
    )
    print("   ✓ Multiple candidate outputs are valid scores")

    print("\n✅ All call method tests passed!")


def test_naml_compilation(naml_model, batch_size=2):
    """Test model compilation and training step."""
    print("\n" + "=" * 50)
    print("TESTING MODEL COMPILATION AND TRAINING")
    print("=" * 50)

    # Compile the model
    print("\n1. Compiling model...")
    naml_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.CategoricalAccuracy()],
    )
    print("   ✓ Model compiled successfully")

    # Test a training step
    print("\n2. Testing training step...")

    # Prepare inputs with separate keys as expected by NAML
    x = {
        "hist_tokens": np.random.randint(0, 1000, (batch_size, 50, 30), dtype=np.int32),
        "hist_abstract_tokens": np.random.randint(
            0, 1000, (batch_size, 50, 100), dtype=np.int32
        ),
        "hist_category": np.random.randint(0, 20, (batch_size, 50), dtype=np.int32),
        "hist_subcategory": np.random.randint(0, 50, (batch_size, 50), dtype=np.int32),
        "cand_tokens": np.random.randint(0, 1000, (batch_size, 5, 30), dtype=np.int32),
        "cand_abstract_tokens": np.random.randint(
            0, 1000, (batch_size, 5, 100), dtype=np.int32
        ),
        "cand_category": np.random.randint(0, 20, (batch_size, 5), dtype=np.int32),
        "cand_subcategory": np.random.randint(0, 50, (batch_size, 5), dtype=np.int32),
    }

    # Create labels
    y = np.zeros((batch_size, 5), dtype=np.float32)
    y[:, 0] = 1.0  # First item is positive

    # Perform one training step
    loss = naml_model.train_on_batch(x, y)
    print(f"   ✓ Training step completed, loss: {loss}")

    # Test evaluation
    print("\n3. Testing evaluation...")
    eval_results = naml_model.evaluate(x, y, verbose=0)
    print(f"   ✓ Evaluation completed, metrics: {eval_results}")

    print("\n✅ Compilation and training tests passed!")


def test_naml_model():
    """Comprehensive test of NAML model."""

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
        "category_embedding_dim": 100,
        "subcategory_embedding_dim": 100,
        "cnn_filter_num": 400,
        "cnn_kernel_size": 3,
        "word_attention_query_dim": 200,
        "view_attention_query_dim": 200,
        "user_attention_query_dim": 200,
        "dropout_rate": 0.2,
        "activation": "relu",
        "max_history_length": 50,
        "max_impressions_length": 5,
        "seed": 42,
    }

    try:
        # Create NAML model
        print("=" * 50)
        print("CREATING NAML MODEL")
        print("=" * 50)
        naml_model = NAML(**model_params)
        print("✓ NAML model created successfully")

        # Check that model is built
        assert naml_model.news_encoder is not None, "News encoder not initialized"
        assert naml_model.user_encoder is not None, "User encoder not initialized"
        assert naml_model.title_encoder is not None, "Title encoder not initialized"
        assert naml_model.abstract_encoder is not None, (
            "Abstract encoder not initialized"
        )
        assert naml_model.category_encoder is not None, (
            "Category encoder not initialized"
        )
        assert naml_model.subcategory_encoder is not None, (
            "Subcategory encoder not initialized"
        )
        assert naml_model.scorer is not None, "Scorer not initialized"
        assert naml_model.training_model is not None, "Training model not initialized"
        assert naml_model.scorer_model is not None, "Scorer model not initialized"
        print("✓ All model components initialized")

        # Run component tests
        test_naml_components(naml_model, batch_size=2)

        # Run call method tests
        test_naml_call_method(naml_model, batch_size=2)

        # Run compilation and training tests
        test_naml_compilation(naml_model, batch_size=2)

        # Print model summary
        print("\n" + "=" * 50)
        print("MODEL SUMMARY")
        print("=" * 50)
        print("\nMain NAML Model:")
        naml_model.summary()

        print("\n🎉 All NAML tests passed successfully!")

    except Exception as e:
        print(f"\n❌ Error testing NAML model: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    keras.utils.set_random_seed(42)

    print("\n" + "=" * 60)
    print("NAML MODEL COMPREHENSIVE TEST SUITE")
    print(f"Backend: {keras.backend.backend()}")
    print("=" * 60)

    success = test_naml_model()

    if success:
        print("\n" + "=" * 60)
        print("✅ NAML MODEL IS FULLY FUNCTIONAL AND READY FOR TRAINING!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ NAML MODEL IMPLEMENTATION NEEDS FIXES")
        print("=" * 60)
