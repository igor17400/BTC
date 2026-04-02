#!/usr/bin/env python3
"""Diagnostic script to check GloVe embeddings loading issue."""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set backend before importing keras
os.environ["KERAS_BACKEND"] = "jax"

from src.core.data.loaders.cache import CacheManager
from src.core.data.encoders.embeddings import EmbeddingsManager
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def diagnose_glove_loading():
    """Diagnose GloVe embeddings loading issue."""
    
    # Initialize cache manager with default .cache directory
    cache_manager = CacheManager()  # Uses default .cache directory
    logger.info(f"Cache directory: {cache_manager.cache_dir}")
    
    # Check if GloVe directory exists
    glove_path = cache_manager.get_embedding_path("glove", 300)
    logger.info(f"GloVe path: {glove_path}")
    logger.info(f"GloVe path exists: {glove_path.exists()}")
    
    if glove_path.exists():
        # List files in GloVe directory
        files = list(glove_path.iterdir())
        logger.info(f"Files in GloVe directory: {[f.name for f in files]}")
        
        # Check for specific files
        txt_file = glove_path / "glove.840B.300d.txt"
        npy_file = glove_path / "glove.840B.300d.npy"
        zip_file = glove_path / "glove.840B.300d.zip"
        
        logger.info(f"Text file exists: {txt_file.exists()} (size: {txt_file.stat().st_size if txt_file.exists() else 'N/A'})")
        logger.info(f"NPY file exists: {npy_file.exists()} (size: {npy_file.stat().st_size if npy_file.exists() else 'N/A'})")
        logger.info(f"ZIP file exists: {zip_file.exists()} (size: {zip_file.stat().st_size if zip_file.exists() else 'N/A'})")
    
    # Try to initialize embeddings manager
    logger.info("\nInitializing EmbeddingsManager...")
    embeddings_manager = EmbeddingsManager(cache_manager)
    
    # Try to load GloVe embeddings
    logger.info("\nAttempting to load GloVe embeddings...")
    try:
        glove_tensor, vocab_map = embeddings_manager.load_glove_embeddings_tf_and_vocab_map(300)
        
        if glove_tensor is not None and vocab_map is not None:
            logger.info(f"✓ Successfully loaded GloVe embeddings!")
            logger.info(f"  Tensor shape: {glove_tensor.shape}")
            logger.info(f"  Vocabulary size: {len(vocab_map)}")
            logger.info(f"  Sample words: {list(vocab_map.keys())[:10]}")
        else:
            logger.error("✗ Failed to load GloVe embeddings - returned None")
            
            # Check if embeddings dict was loaded
            if embeddings_manager.glove_embeddings is not None:
                logger.info(f"  But glove_embeddings dict exists with {len(embeddings_manager.glove_embeddings)} entries")
            else:
                logger.error("  glove_embeddings dict is None")
                
    except Exception as e:
        logger.error(f"✗ Exception while loading GloVe embeddings: {e}")
        import traceback
        traceback.print_exc()
    
    # Check memory
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    logger.info(f"\nMemory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
    
    # Check available system memory
    mem = psutil.virtual_memory()
    logger.info(f"System memory: {mem.available / 1024 / 1024:.2f} MB available of {mem.total / 1024 / 1024:.2f} MB total")

if __name__ == "__main__":
    diagnose_glove_loading()