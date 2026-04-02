#!/usr/bin/env python3
"""Comprehensive test script for LSTUR model implementation."""

import os
import sys
from pathlib import Path

# Set JAX as Keras backend before importing keras
os.environ["KERAS_BACKEND"] = "jax"

# Add the project root to Python path to enable imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import keras
from src.frameworks.keras.models.lstur import LSTUR


def create_dummy_data(batch_size=2, num_users=1000):
    """Create dummy data for testing LSTUR."""
    data = {
        # Basic inputs (title tokens)
        "hist_tokens": np.random.randint(1, 1000, (batch_size, 50, 50), dtype=np.int32),
        "cand_tokens": np.random.randint(1, 1000, (batch_size, 5, 50), dtype=np.int32),
        "single_candidate_tokens": np.random.randint(1, 1000, (batch_size, 50), dtype=np.int32),
        
        # User IDs (required for LSTUR)
        "user_ids_2d": np.random.randint(0, num_users, (batch_size, 1), dtype=np.int32),
        "user_ids_1d": np.random.randint(0, num_users, (batch_size,), dtype=np.int32),
        
        # Labels for training
        "labels": np.zeros((batch_size, 5), dtype=np.float32),
    }
    # Set first item as positive for each batch
    data["labels"][:, 0] = 1.0
    return data


def test_lstur_components(lstur_model, batch_size=2):
    """Test individual LSTUR components."""
    print("\n" + "="*50)
    print("TESTING INDIVIDUAL COMPONENTS")
    print("="*50)
    
    dummy_data = create_dummy_data(batch_size)
    
    # Test news encoder
    print("\n1. Testing news encoder...")
    news_input = np.random.randint(1, 1000, (batch_size, 50), dtype=np.int32)
    news_output = lstur_model.news_encoder(news_input, training=False)
    print(f"   ✓ News encoder output shape: {news_output.shape}")
    expected_dim = lstur_model.config.cnn_filter_num
    assert news_output.shape == (batch_size, expected_dim), f"Expected shape (batch_size, {expected_dim}), got {news_output.shape}"
    
    # Test user encoder with 2D user IDs
    print("\n2. Testing user encoder with 2D user IDs...")
    user_output_2d = lstur_model.user_encoder([dummy_data["hist_tokens"], dummy_data["user_ids_2d"]], training=False)
    print(f"   ✓ User encoder (2D user IDs) output shape: {user_output_2d.shape}")
    expected_dim = lstur_model.config.gru_unit
    assert user_output_2d.shape == (batch_size, expected_dim), f"Expected shape (batch_size, {expected_dim}), got {user_output_2d.shape}"
    
    # Test user encoder with 1D user IDs (for fast evaluation)
    print("\n3. Testing user encoder with 1D user IDs...")
    user_output_1d = lstur_model.user_encoder([dummy_data["hist_tokens"], dummy_data["user_ids_1d"]], training=False)
    print(f"   ✓ User encoder (1D user IDs) output shape: {user_output_1d.shape}")
    assert user_output_1d.shape == (batch_size, expected_dim), f"Expected shape (batch_size, {expected_dim}), got {user_output_1d.shape}"
    
    # Test scorer component
    print("\n4. Testing scorer component...")
    
    # Test training batch scoring
    training_scores = lstur_model.scorer.score_training_batch(
        dummy_data["hist_tokens"], 
        dummy_data["user_ids_2d"], 
        dummy_data["cand_tokens"], 
        training=False
    )
    print(f"   ✓ Training batch scores shape: {training_scores.shape}")
    assert training_scores.shape == (batch_size, 5), f"Expected shape (batch_size, 5), got {training_scores.shape}"
    
    # Verify softmax output (should sum to 1)
    score_sums = np.sum(training_scores, axis=-1)
    assert np.allclose(score_sums, 1.0, rtol=1e-5), f"Softmax scores don't sum to 1: {score_sums}"
    print(f"   ✓ Softmax scores sum to 1.0: {score_sums}")
    
    # Test single candidate scoring
    single_scores = lstur_model.scorer.score_single_candidate(
        dummy_data["hist_tokens"], 
        dummy_data["user_ids_2d"], 
        dummy_data["single_candidate_tokens"], 
        training=False
    )
    print(f"   ✓ Single candidate scores shape: {single_scores.shape}")
    assert single_scores.shape == (batch_size, 1), f"Expected shape (batch_size, 1), got {single_scores.shape}"
    
    # Verify sigmoid output
    assert np.all(single_scores >= 0) and np.all(single_scores <= 1), "Sigmoid outputs should be in [0, 1]"
    print(f"   ✓ Single candidate scores are valid sigmoid outputs")
    
    # Test multiple candidates scoring
    multi_scores = lstur_model.scorer.score_multiple_candidates(
        dummy_data["hist_tokens"], 
        dummy_data["user_ids_2d"], 
        dummy_data["cand_tokens"], 
        training=False
    )
    print(f"   ✓ Multiple candidates scores shape: {multi_scores.shape}")
    assert multi_scores.shape == (batch_size, 5), f"Expected shape (batch_size, 5), got {multi_scores.shape}"
    
    print("\n✅ All component tests passed!")


def test_lstur_call_method(lstur_model, batch_size=2):
    """Test the main call method with different input formats."""
    print("\n" + "="*50)
    print("TESTING MAIN CALL METHOD")
    print("="*50)
    
    dummy_data = create_dummy_data(batch_size)
    
    # Test 1: Training mode
    print("\n1. Testing training mode...")
    training_inputs = {
        "hist_tokens": dummy_data["hist_tokens"],
        "user_ids": dummy_data["user_ids_2d"],
        "cand_tokens": dummy_data["cand_tokens"]
    }
    
    training_output = lstur_model(training_inputs, training=True)
    print(f"   ✓ Training output shape: {training_output.shape}")
    assert training_output.shape == (batch_size, 5), f"Expected shape (batch_size, 5), got {training_output.shape}"
    
    # Verify softmax output
    score_sums = np.sum(training_output, axis=-1)
    assert np.allclose(score_sums, 1.0, rtol=1e-5), f"Softmax scores don't sum to 1: {score_sums}"
    print(f"   ✓ Training outputs are valid softmax scores")
    
    # Test 2: Inference mode with single candidate
    print("\n2. Testing inference mode with single candidate...")
    single_cand_inputs = {
        "hist_tokens": dummy_data["hist_tokens"],
        "user_ids": dummy_data["user_ids_2d"],
        "single_candidate_tokens": dummy_data["single_candidate_tokens"]
    }
    
    single_output = lstur_model(single_cand_inputs, training=False)
    print(f"   ✓ Single candidate output shape: {single_output.shape}")
    assert single_output.shape == (batch_size, 1), f"Expected shape (batch_size, 1), got {single_output.shape}"
    
    # Verify sigmoid output
    assert np.all(single_output >= 0) and np.all(single_output <= 1), "Sigmoid outputs should be in [0, 1]"
    print(f"   ✓ Single candidate outputs are valid sigmoid scores")
    
    # Test 3: Inference mode with multiple candidates
    print("\n3. Testing inference mode with multiple candidates...")
    multi_cand_inputs = {
        "hist_tokens": dummy_data["hist_tokens"],
        "user_ids": dummy_data["user_ids_2d"],
        "cand_tokens": dummy_data["cand_tokens"]
    }
    
    multi_output = lstur_model(multi_cand_inputs, training=False)
    print(f"   ✓ Multiple candidates output shape: {multi_output.shape}")
    assert multi_output.shape == (batch_size, 5), f"Expected shape (batch_size, 5), got {multi_output.shape}"
    
    # Verify sigmoid outputs
    assert np.all(multi_output >= 0) and np.all(multi_output <= 1), "Sigmoid outputs should be in [0, 1]"
    print(f"   ✓ Multiple candidate outputs are valid sigmoid scores")
    
    print("\n✅ All call method tests passed!")


def test_lstur_compatibility_models(lstur_model, batch_size=2):
    """Test the compatibility models (training_model and scorer_model)."""
    print("\n" + "="*50)
    print("TESTING COMPATIBILITY MODELS")
    print("="*50)
    
    dummy_data = create_dummy_data(batch_size)
    
    # Test training_model
    print("\n1. Testing training_model...")
    training_output = lstur_model.training_model([
        dummy_data["hist_tokens"],
        dummy_data["user_ids_2d"],
        dummy_data["cand_tokens"]
    ])
    print(f"   ✓ Training model output shape: {training_output.shape}")
    assert training_output.shape == (batch_size, 5), f"Expected shape (batch_size, 5), got {training_output.shape}"
    
    # Verify softmax output
    score_sums = np.sum(training_output, axis=-1)
    assert np.allclose(score_sums, 1.0, rtol=1e-5), f"Softmax scores don't sum to 1: {score_sums}"
    print(f"   ✓ Training model outputs are valid softmax scores")
    
    # Test scorer_model
    print("\n2. Testing scorer_model...")
    scoring_output = lstur_model.scorer_model([
        dummy_data["hist_tokens"],
        dummy_data["user_ids_2d"],
        dummy_data["single_candidate_tokens"]
    ])
    print(f"   ✓ Scorer model output shape: {scoring_output.shape}")
    assert scoring_output.shape == (batch_size, 1), f"Expected shape (batch_size, 1), got {scoring_output.shape}"
    
    # Verify sigmoid output
    assert np.all(scoring_output >= 0) and np.all(scoring_output <= 1), "Sigmoid outputs should be in [0, 1]"
    print(f"   ✓ Scorer model outputs are valid sigmoid scores")
    
    print("\n✅ All compatibility model tests passed!")


def test_lstur_encoder_types(batch_size=2):
    """Test both 'ini' and 'con' encoder types."""
    print("\n" + "="*50)
    print("TESTING ENCODER TYPES")
    print("="*50)
    
    # Mock processed news data
    processed_news = {
        "vocab_size": 10000,
        "num_categories": 20,
        "num_subcategories": 50,
        "embeddings": np.random.randn(10000, 300).astype(np.float32)
    }
    
    dummy_data = create_dummy_data(batch_size)
    
    # Test 'ini' type (default)
    print("\n1. Testing 'ini' encoder type...")
    lstur_ini = LSTUR(
        processed_news=processed_news,
        num_users=1000,
        type="ini",
        embedding_size=300,
        cnn_filter_num=300,
        gru_unit=300,
        max_title_length=50,
        max_history_length=50,
        max_impressions_length=5,
        process_user_id=True
    )
    
    user_output_ini = lstur_ini.user_encoder([dummy_data["hist_tokens"], dummy_data["user_ids_2d"]], training=False)
    print(f"   ✓ 'ini' encoder output shape: {user_output_ini.shape}")
    assert user_output_ini.shape == (batch_size, 300), f"Expected shape (batch_size, 300), got {user_output_ini.shape}"
    
    # Test 'con' type
    print("\n2. Testing 'con' encoder type...")
    lstur_con = LSTUR(
        processed_news=processed_news,
        num_users=1000,
        type="con",
        embedding_size=300,
        cnn_filter_num=300,
        gru_unit=300,
        max_title_length=50,
        max_history_length=50,
        max_impressions_length=5,
        process_user_id=True
    )
    
    user_output_con = lstur_con.user_encoder([dummy_data["hist_tokens"], dummy_data["user_ids_2d"]], training=False)
    print(f"   ✓ 'con' encoder output shape: {user_output_con.shape}")
    assert user_output_con.shape == (batch_size, 300), f"Expected shape (batch_size, 300), got {user_output_con.shape}"
    
    print("\n✅ All encoder type tests passed!")


def test_lstur_compilation(lstur_model, batch_size=2):
    """Test model compilation and training step."""
    print("\n" + "="*50)
    print("TESTING MODEL COMPILATION AND TRAINING")
    print("="*50)
    
    # Compile the model
    print("\n1. Compiling model...")
    lstur_model.compile(
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
        "user_ids": dummy_data["user_ids_2d"],
        "cand_tokens": dummy_data["cand_tokens"]
    }
    y = dummy_data["labels"]
    
    # Perform one training step
    loss = lstur_model.train_on_batch(x, y)
    print(f"   ✓ Training step completed, loss: {loss}")
    
    # Test evaluation
    print("\n3. Testing evaluation...")
    eval_results = lstur_model.evaluate(x, y, verbose=0)
    print(f"   ✓ Evaluation completed, metrics: {eval_results}")
    
    print("\n✅ Compilation and training tests passed!")


def test_lstur_model():
    """Comprehensive test of LSTUR model."""
    
    # Mock processed news data
    processed_news = {
        "vocab_size": 10000,
        "num_categories": 20,
        "num_subcategories": 50,
        "embeddings": np.random.randn(10000, 300).astype(np.float32)
    }
    
    # Model parameters
    model_params = {
        "processed_news": processed_news,
        "num_users": 1000,
        "embedding_size": 300,
        "cnn_filter_num": 300,
        "cnn_kernel_size": 3,
        "cnn_activation": "relu",
        "attention_hidden_dim": 200,
        "gru_unit": 300,
        "type": "ini",
        "dropout_rate": 0.2,
        "seed": 42,
        "max_title_length": 50,
        "max_history_length": 50,
        "max_impressions_length": 5,
        "process_user_id": True
    }
    
    try:
        # Create LSTUR model
        print("="*50)
        print("CREATING LSTUR MODEL")
        print("="*50)
        lstur_model = LSTUR(**model_params)
        print("✓ LSTUR model created successfully")
        
        # Check that model is built
        assert lstur_model.news_encoder is not None, "News encoder not initialized"
        assert lstur_model.user_encoder is not None, "User encoder not initialized"
        assert lstur_model.scorer is not None, "Scorer not initialized"
        assert lstur_model.training_model is not None, "Training model not initialized"
        assert lstur_model.scorer_model is not None, "Scorer model not initialized"
        print("✓ All model components initialized")
        
        # Run component tests
        test_lstur_components(lstur_model, batch_size=2)
        
        # Run call method tests
        test_lstur_call_method(lstur_model, batch_size=2)
        
        # Run compatibility model tests
        test_lstur_compatibility_models(lstur_model, batch_size=2)
        
        # Run encoder type tests
        test_lstur_encoder_types(batch_size=2)
        
        # Run compilation and training tests
        test_lstur_compilation(lstur_model, batch_size=2)
        
        # Print model summary
        print("\n" + "="*50)
        print("MODEL SUMMARY")
        print("="*50)
        print("\nMain LSTUR Model:")
        lstur_model.summary()
        
        print("\n" + "="*50)
        print("MODEL CONFIGURATION")
        print("="*50)
        print(f"Config: {lstur_model.config}")
        print(f"Number of users: {lstur_model.num_users}")
        print(f"News encoder output dimension: {lstur_model.config.cnn_filter_num}")
        print(f"User encoder output dimension: {lstur_model.config.gru_unit}")
        print(f"User encoder type: {lstur_model.config.type}")
        print(f"Process user ID: {lstur_model.config.process_user_id}")
        
        print("\n🎉 All LSTUR tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error testing LSTUR model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    keras.utils.set_random_seed(42)
    
    print("\n" + "="*60)
    print("LSTUR MODEL COMPREHENSIVE TEST SUITE")
    print(f"Backend: {keras.backend.backend()}")
    print("="*60)
    
    success = test_lstur_model()
    
    if success:
        print("\n" + "="*60)
        print("✅ LSTUR MODEL IS FULLY FUNCTIONAL AND READY FOR TRAINING!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ LSTUR MODEL IMPLEMENTATION NEEDS FIXES")
        print("="*60)