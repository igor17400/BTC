import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("bpemb_manager")


class BPEmbManager:
    """Manager for BPEmb multilingual embeddings with systematic download and caching."""

    def __init__(self, cache_manager=None, float_dtype: str = "float32"):
        self.cache_manager = cache_manager
        self.bpemb_models: dict[str, object] = {}
        self.supported_languages = self._get_supported_languages()
        self.float_dtype = float_dtype

    def _get_supported_languages(self) -> list[str]:
        """Get list of supported BPEmb languages.

        Full list available at: https://bpemb.h-its.org/
        """
        # fmt: off
        return [
            # Major world languages
            "en", "de", "fr", "es", "it", "pt", "ru", "ja", "ko", "zh",
            "ar", "hi", "tr", "pl", "nl", "sv", "da", "no", "fi", "cs",
            # Central/Eastern European
            "sk", "hu", "hr", "sr", "bg", "uk", "ro", "sl", "et", "lv",
            "lt", "sq", "mk", "el",
            # Celtic, Basque, Insular
            "mt", "cy", "ga", "is", "fo", "kl", "eu",
            # Romance (regional)
            "ca", "gl", "ast", "an", "oc", "co", "sc", "rm", "fur", "lld",
            "vec", "lij", "pms", "lmo", "rgn", "nap", "scn", "srd", "egl",
            # Creoles & Pidgins
            "ht", "pap", "gcf", "acf", "bzs", "jam", "tcs", "bi", "tpi",
            # African
            "sw", "lg", "ak", "tw", "bm", "dyu", "ff", "fuv", "wo", "sg",
            "ln", "kg", "lua", "luo", "luy", "nyn", "cgg", "teo", "lgg",
            "ach", "laj", "mas", "saq", "kam", "mer", "ebu", "guz", "kln",
            "tn", "ts", "ve", "xh", "zu", "af", "nso", "st", "ss", "nr",
            # South Asian
            "as", "bn", "gu", "kn", "ml", "mr", "or", "pa", "ta", "te",
            "ur", "sd", "ks", "ne", "si", "bho", "mai", "new", "bpy",
            "gom", "kok",
            # Southeast Asian
            "my", "km", "lo", "th", "vi", "ms", "id", "jv", "su",
            "tl", "ceb", "ilo", "war", "pag", "bcl", "hil",
            # Central Asian & Turkic
            "kk", "ky", "uz", "tk", "tt", "ba", "cv", "sah", "tyv",
            "kjh", "alt", "nog",
            # East Asian (regional)
            "mn", "bo", "dz", "ug", "yue", "wuu", "hsn", "nan",
            "hak", "cdo", "gan",
            # Horn of Africa & Semitic
            "am", "ti", "om", "so", "aa",
            # Germanic (historical & regional)
            "nds", "frr", "stq", "fy", "li", "vls", "zea", "ksh", "bar",
            "lb", "gsw", "als", "wae",
            # Baltic
            "ltg", "prg",
            # Uralic
            "vot", "vep", "krl", "olo", "myv", "mdf", "kpv", "koi",
            "udm", "mhr", "mrj", "kv",
            # Siberian & Mongolic
            "sel", "yrk", "eve", "evn", "bua", "xwo",
            # Misc
            "kea", "krj", "knf", "mww", "za", "ii",
        ]
        # fmt: on

    def is_language_supported(self, language_code: str) -> bool:
        """Check if a language is supported by BPEmb."""
        return language_code.lower() in self.supported_languages

    def install_bpemb(self) -> bool:
        """Install BPEmb library if not available."""
        import importlib.util

        if importlib.util.find_spec("bpemb") is not None:
            logger.info("BPEmb library is already available")
            return True
        else:
            logger.info("BPEmb library not found, attempting to install...")
            try:
                import subprocess
                import sys

                subprocess.check_call([sys.executable, "-m", "pip", "install", "bpemb"])
                logger.info("Successfully installed BPEmb")
                return True
            except Exception as e:
                logger.error(f"Failed to install BPEmb: {e}")
                return False

    def load_bpemb_model(
        self,
        language: str,
        vocab_size: int = 10000,
        embedding_dim: int = 300,
        cache_dir: str | None = None,
    ) -> object | None:
        """
        Load a BPEmb model for the specified language.

        Args:
            language: Language code (e.g., 'ja', 'de', 'fr')
            vocab_size: Vocabulary size (typically 1000, 3000, 5000, 10000, 25000, 50000, 100000)
            embedding_dim: Embedding dimension (typically 25, 50, 100, 200, 300)
            cache_dir: Directory to cache the model

        Returns:
            BPEmb model object or None if loading failed
        """
        if not self.install_bpemb():
            return None

        # Normalize language code
        lang_code = language.lower()
        if not self.is_language_supported(lang_code):
            logger.warning(
                f"Language '{lang_code}' might not be supported by BPEmb. Available languages: {len(self.supported_languages)} total"
            )
            # Still try to load - BPEmb might support it

        model_key = f"{lang_code}_{vocab_size}_{embedding_dim}"

        # Check if model is already loaded
        if model_key in self.bpemb_models:
            logger.info(f"Using cached BPEmb model for {model_key}")
            return self.bpemb_models[model_key]

        try:
            from bpemb import BPEmb

            # Set cache directory if using cache manager
            if cache_dir is None and self.cache_manager:
                cache_dir = str(
                    self.cache_manager.get_embedding_path("bpemb", lang_code)
                )
                Path(cache_dir).mkdir(parents=True, exist_ok=True)

            logger.info(
                f"Loading BPEmb model: {lang_code}, vocab_size={vocab_size}, dim={embedding_dim}"
            )

            # Load the model - BPEmb will download automatically if needed
            bpemb_model = BPEmb(
                lang=lang_code, vs=vocab_size, dim=embedding_dim, cache_dir=cache_dir
            )

            # Cache the model
            self.bpemb_models[model_key] = bpemb_model

            logger.info(f"Successfully loaded BPEmb model for {lang_code}")
            logger.info(f"Model vocabulary size: {len(bpemb_model.words)}")
            logger.info(f"Embedding shape: {bpemb_model.vectors.shape}")

            return bpemb_model

        except Exception as e:
            logger.error(f"Failed to load BPEmb model for {lang_code}: {e}")
            return None

    def get_bpemb_embeddings_and_vocab(
        self,
        language: str,
        vocab_size: int = 10000,
        embedding_dim: int = 300,
        max_vocab_size: int | None = None,
    ) -> tuple[np.ndarray | None, dict[str, int] | None]:
        """
        Get BPEmb embeddings as numpy array and vocabulary mapping.

        Args:
            language: Language code
            vocab_size: BPEmb vocabulary size
            embedding_dim: Embedding dimension
            max_vocab_size: Maximum vocabulary size to return (for memory management)

        Returns:
            Tuple of (embedding_matrix, word_to_index_mapping)
        """
        model = self.load_bpemb_model(language, vocab_size, embedding_dim)
        if model is None:
            return None, None

        try:
            # Get embeddings and vocabulary
            embeddings = (
                model.vectors
            )  # numpy array of shape (vocab_size, embedding_dim)
            words = model.words  # list of words/subwords

            # Limit vocabulary size if specified
            if max_vocab_size and len(words) > max_vocab_size:
                logger.info(
                    f"Limiting vocabulary from {len(words)} to {max_vocab_size} most frequent terms"
                )
                embeddings = embeddings[:max_vocab_size]
                words = words[:max_vocab_size]

            # Create word to index mapping
            word_to_idx = {word: idx for idx, word in enumerate(words)}

            logger.info(f"Retrieved BPEmb embeddings: {embeddings.shape}")

            return embeddings.astype(self.float_dtype), word_to_idx

        except Exception as e:
            logger.error(f"Failed to extract embeddings from BPEmb model: {e}")
            return None, None

    def encode_text(
        self,
        text: str,
        language: str,
        vocab_size: int = 10000,
        embedding_dim: int = 300,
    ) -> list[str] | None:
        """
        Encode text into BPE subwords.

        Args:
            text: Input text to encode
            language: Language code
            vocab_size: BPEmb vocabulary size
            embedding_dim: Embedding dimension

        Returns:
            List of subword tokens
        """
        model = self.load_bpemb_model(language, vocab_size, embedding_dim)
        if model is None:
            return None

        try:
            return model.encode(text)
        except Exception as e:
            logger.error(f"Failed to encode text: {e}")
            return None

    def embed_text(
        self,
        text: str,
        language: str,
        vocab_size: int = 10000,
        embedding_dim: int = 300,
    ) -> np.ndarray | None:
        """
        Embed text into vector representation.

        Args:
            text: Input text to embed
            language: Language code
            vocab_size: BPEmb vocabulary size
            embedding_dim: Embedding dimension

        Returns:
            Numpy array of embeddings for each subword
        """
        model = self.load_bpemb_model(language, vocab_size, embedding_dim)
        if model is None:
            return None

        try:
            embeddings = model.embed(text)
            return np.array(embeddings, dtype=self.float_dtype)
        except Exception as e:
            logger.error(f"Failed to embed text: {e}")
            return None

    def get_language_code_mapping(self) -> dict[str, str]:
        """Get mapping from common language names to BPEmb codes."""
        return {
            "english": "en",
            "japanese": "ja",
            "german": "de",
            "french": "fr",
            "spanish": "es",
            "italian": "it",
            "portuguese": "pt",
            "russian": "ru",
            "korean": "ko",
            "chinese": "zh",
            "arabic": "ar",
            "hindi": "hi",
            "turkish": "tr",
            "polish": "pl",
            "dutch": "nl",
            "swedish": "sv",
            "danish": "da",
            "norwegian": "no",
            "finnish": "fi",
            "czech": "cs",
            "slovak": "sk",
            "hungarian": "hu",
            "croatian": "hr",
            "serbian": "sr",
            "bulgarian": "bg",
            "ukrainian": "uk",
            "romanian": "ro",
            "slovenian": "sl",
            "estonian": "et",
            "latvian": "lv",
            "lithuanian": "lt",
        }

    def get_language_code(self, language_name: str) -> str:
        """Convert language name to BPEmb language code."""
        mapping = self.get_language_code_mapping()
        return mapping.get(language_name.lower(), language_name.lower())

    def create_filtered_embedding_matrix(
        self,
        vocab: dict[str, int],
        language: str,
        embedding_dim: int = 300,
        vocab_size: int = 10000,
    ) -> np.ndarray:
        """
        Create an embedding matrix filtered to match a specific vocabulary.

        Args:
            vocab: Target vocabulary (word -> index mapping)
            language: Language code
            embedding_dim: Embedding dimension
            vocab_size: BPEmb vocabulary size

        Returns:
            Embedding matrix of shape (len(vocab), embedding_dim)
        """
        # Get BPEmb embeddings
        bpemb_embeddings, bpemb_vocab = self.get_bpemb_embeddings_and_vocab(
            language, vocab_size, embedding_dim
        )

        if bpemb_embeddings is None or bpemb_vocab is None:
            logger.warning(
                f"Could not load BPEmb embeddings for {language}, using random initialization"
            )
            # Fallback to random embeddings
            return (
                np.random.randn(len(vocab), embedding_dim).astype(self.float_dtype)
                * 0.1
            )

        # Initialize embedding matrix
        embedding_matrix = (
            np.random.randn(len(vocab), embedding_dim).astype(self.float_dtype) * 0.1
        )

        # Fill in embeddings for words that exist in BPEmb
        found_words = 0
        for word, idx in vocab.items():
            if word in bpemb_vocab:
                bpemb_idx = bpemb_vocab[word]
                embedding_matrix[idx] = bpemb_embeddings[bpemb_idx]
                found_words += 1

        logger.info(
            f"Found BPEmb embeddings for {found_words}/{len(vocab)} vocabulary words ({found_words / len(vocab) * 100:.1f}%)"
        )

        return embedding_matrix
