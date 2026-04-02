"""
Comprehensive test suite for validating Japanese and English embeddings.
Tests embedding quality, consistency, and expected properties.
"""

import numpy as np
import os
from typing import Any
from scipy.spatial.distance import cosine
from scipy.stats import normaltest, kurtosis, skew
import warnings

class EmbeddingValidator:
    """Validates embedding matrices for quality and consistency."""
    
    def __init__(self, embedding_path: str, language: str = "unknown"):
        """
        Initialize validator with embedding file.
        
        Args:
            embedding_path: Path to .npy embedding file
            language: Language identifier (e.g., 'japanese', 'english')
        """
        self.embedding_path = embedding_path
        self.language = language
        self.embedding_matrix = None
        self.vocab_size = None
        self.embedding_dim = None
        
    def load_embeddings(self) -> bool:
        """Load embeddings from file."""
        try:
            self.embedding_matrix = np.load(self.embedding_path)
            self.vocab_size, self.embedding_dim = self.embedding_matrix.shape
            return True
        except Exception as e:
            print(f"Failed to load embeddings: {e}")
            return False
            
    def test_basic_properties(self) -> dict[str, Any]:
        """Test basic statistical properties of embeddings."""
        results = {
            "shape": self.embedding_matrix.shape,
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "min_value": np.min(self.embedding_matrix),
            "max_value": np.max(self.embedding_matrix),
            "mean_value": np.mean(self.embedding_matrix),
            "std_deviation": np.std(self.embedding_matrix),
            "has_nans": np.isnan(self.embedding_matrix).any(),
            "has_infs": np.isinf(self.embedding_matrix).any(),
            "num_zeros": np.sum(self.embedding_matrix == 0),
            "percent_zeros": 100 * np.sum(self.embedding_matrix == 0) / self.embedding_matrix.size
        }
        return results
        
    def test_distribution(self) -> dict[str, Any]:
        """Test if embeddings follow expected distribution (roughly normal)."""
        flat_embeddings = self.embedding_matrix.flatten()
        
        # Remove zeros for distribution testing (padding tokens)
        non_zero_embeddings = flat_embeddings[flat_embeddings != 0]
        
        results = {}
        
        # Normality test
        if len(non_zero_embeddings) > 0:
            stat, p_value = normaltest(non_zero_embeddings[:10000])  # Sample for speed
            results["normality_p_value"] = p_value
            results["is_roughly_normal"] = p_value > 0.05
            results["skewness"] = skew(non_zero_embeddings)
            results["kurtosis"] = kurtosis(non_zero_embeddings)
        
        # Check if centered around 0
        results["mean_near_zero"] = abs(np.mean(non_zero_embeddings)) < 0.1
        
        return results
        
    def test_vector_norms(self) -> dict[str, Any]:
        """Test vector norm properties."""
        norms = np.linalg.norm(self.embedding_matrix, axis=1)
        
        # Exclude padding (zero vectors)
        non_zero_norms = norms[norms > 1e-8]
        
        results = {
            "min_norm": np.min(non_zero_norms) if len(non_zero_norms) > 0 else 0,
            "max_norm": np.max(non_zero_norms) if len(non_zero_norms) > 0 else 0,
            "mean_norm": np.mean(non_zero_norms) if len(non_zero_norms) > 0 else 0,
            "std_norm": np.std(non_zero_norms) if len(non_zero_norms) > 0 else 0,
            "num_zero_vectors": np.sum(norms < 1e-8),
            "percent_zero_vectors": 100 * np.sum(norms < 1e-8) / len(norms)
        }
        
        # Check if norms are reasonable (typically between 0.5 and 5)
        if len(non_zero_norms) > 0:
            results["norms_in_reasonable_range"] = (
                (results["min_norm"] > 0.1) and (results["max_norm"] < 10)
            )
        
        return results
        
    def test_similarity_distribution(self, sample_size: int = 1000) -> dict[str, Any]:
        """Test cosine similarity distribution between random pairs."""
        results = {}
        
        # Get non-zero vectors
        norms = np.linalg.norm(self.embedding_matrix, axis=1)
        non_zero_indices = np.where(norms > 1e-8)[0]
        
        if len(non_zero_indices) < 2:
            results["error"] = "Not enough non-zero vectors"
            return results
            
        # Sample random pairs
        n_samples = min(sample_size, len(non_zero_indices))
        idx1 = np.random.choice(non_zero_indices, n_samples, replace=True)
        idx2 = np.random.choice(non_zero_indices, n_samples, replace=True)
        
        similarities = []
        for i, j in zip(idx1, idx2):
            if i != j:
                sim = 1 - cosine(self.embedding_matrix[i], self.embedding_matrix[j])
                similarities.append(sim)
                
        similarities = np.array(similarities)
        
        results = {
            "min_similarity": np.min(similarities),
            "max_similarity": np.max(similarities),
            "mean_similarity": np.mean(similarities),
            "std_similarity": np.std(similarities),
            "similarity_near_zero": abs(np.mean(similarities)) < 0.2,
            "good_variance": np.std(similarities) > 0.1
        }
        
        return results
        
    def test_special_tokens(self) -> dict[str, Any]:
        """Test properties of special tokens (usually first few indices)."""
        results = {}
        
        # Check if first token is padding (all zeros)
        if np.allclose(self.embedding_matrix[0], 0):
            results["has_padding_token"] = True
            results["padding_at_index_0"] = True
        else:
            results["has_padding_token"] = False
            
        # Check for UNK token (usually at index 1)
        if self.vocab_size > 1:
            unk_norm = np.linalg.norm(self.embedding_matrix[1])
            results["likely_has_unk_token"] = unk_norm > 0
            
        return results
        
    def validate_all(self) -> dict[str, Any]:
        """Run all validation tests."""
        if not self.load_embeddings():
            return {"error": "Failed to load embeddings"}
            
        print(f"\n{'='*60}")
        print(f"Validating {self.language.upper()} embeddings")
        print(f"Path: {self.embedding_path}")
        print(f"{'='*60}")
        
        all_results = {
            "language": self.language,
            "path": self.embedding_path,
            "basic_properties": self.test_basic_properties(),
            "distribution": self.test_distribution(),
            "vector_norms": self.test_vector_norms(),
            "similarity": self.test_similarity_distribution(),
            "special_tokens": self.test_special_tokens()
        }
        
        # Overall assessment
        all_results["assessment"] = self.assess_quality(all_results)
        
        return all_results
        
    def assess_quality(self, results: dict[str, Any]) -> dict[str, Any]:
        """Provide overall quality assessment."""
        issues = []
        warnings = []
        
        # Check for critical issues
        basic = results["basic_properties"]
        if basic["has_nans"]:
            issues.append("Contains NaN values")
        if basic["has_infs"]:
            issues.append("Contains infinite values")
        if basic["embedding_dim"] < 50:
            warnings.append(f"Low embedding dimension ({basic['embedding_dim']})")
        if basic["embedding_dim"] > 1000:
            warnings.append(f"Very high embedding dimension ({basic['embedding_dim']})")
            
        # Check distribution
        dist = results.get("distribution", {})
        if not dist.get("mean_near_zero", True):
            warnings.append("Mean not centered around zero")
            
        # Check norms
        norms = results["vector_norms"]
        if not norms.get("norms_in_reasonable_range", True):
            warnings.append("Vector norms outside reasonable range")
        if norms["percent_zero_vectors"] > 50:
            warnings.append(f"High percentage of zero vectors ({norms['percent_zero_vectors']:.1f}%)")
            
        # Check similarity
        sim = results["similarity"]
        if not sim.get("good_variance", True):
            warnings.append("Low variance in similarities")
            
        return {
            "likely_valid": len(issues) == 0,
            "quality_score": max(0, 10 - len(issues)*3 - len(warnings)),
            "issues": issues,
            "warnings": warnings
        }
        
    def print_results(self, results: dict[str, Any]):
        """Pretty print validation results."""
        print("\n📊 Basic Properties:")
        basic = results["basic_properties"]
        print(f"  • Shape: {basic['shape']}")
        print(f"  • Vocab size: {basic['vocab_size']:,}")
        print(f"  • Embedding dim: {basic['embedding_dim']}")
        print(f"  • Value range: [{basic['min_value']:.4f}, {basic['max_value']:.4f}]")
        print(f"  • Mean ± Std: {basic['mean_value']:.4f} ± {basic['std_deviation']:.4f}")
        print(f"  • Zero values: {basic['percent_zeros']:.2f}%")
        
        print("\n📈 Distribution Tests:")
        dist = results.get("distribution", {})
        if dist:
            print(f"  • Roughly normal: {dist.get('is_roughly_normal', 'N/A')}")
            print(f"  • Skewness: {dist.get('skewness', 0):.4f}")
            print(f"  • Mean near zero: {dist.get('mean_near_zero', False)}")
        
        print("\n📏 Vector Norms:")
        norms = results["vector_norms"]
        print(f"  • Range: [{norms['min_norm']:.4f}, {norms['max_norm']:.4f}]")
        print(f"  • Mean ± Std: {norms['mean_norm']:.4f} ± {norms['std_norm']:.4f}")
        print(f"  • Zero vectors: {norms['num_zero_vectors']:,} ({norms['percent_zero_vectors']:.1f}%)")
        
        print("\n🔄 Similarity Distribution:")
        sim = results["similarity"]
        if "error" not in sim:
            print(f"  • Range: [{sim['min_similarity']:.4f}, {sim['max_similarity']:.4f}]")
            print(f"  • Mean ± Std: {sim['mean_similarity']:.4f} ± {sim['std_similarity']:.4f}")
        
        print("\n✅ Assessment:")
        assessment = results["assessment"]
        print(f"  • Valid: {assessment['likely_valid']}")
        print(f"  • Quality score: {assessment['quality_score']}/10")
        
        if assessment["issues"]:
            print(f"  • ❌ Critical issues: {', '.join(assessment['issues'])}")
        if assessment["warnings"]:
            print(f"  • ⚠️  Warnings: {', '.join(assessment['warnings'])}")
            
        if assessment["likely_valid"] and assessment["quality_score"] >= 7:
            print("\n✨ Embeddings appear to be valid and of good quality!")
        elif assessment["likely_valid"]:
            print("\n⚠️  Embeddings are valid but may have quality issues.")
        else:
            print("\n❌ Embeddings have critical issues and may be corrupted.")


def main():
    """Main test runner."""
    # Define embedding paths
    embeddings_to_test = [
        {
            "path": "./data/japanese/v1/processed/filtered_embeddings_thresh3_japanese.npy",
            "language": "japanese"
        },
        {
            "path": "./data/mind/small/processed/filtered_embeddings_thresh5_english.npy",
            "language": "english"
        }
    ]
    
    all_results = {}
    
    for embedding_info in embeddings_to_test:
        path = embedding_info["path"]
        language = embedding_info["language"]
        
        if not os.path.exists(path):
            print(f"\n⚠️  Embedding file not found: {path}")
            continue
            
        validator = EmbeddingValidator(path, language)
        results = validator.validate_all()
        validator.print_results(results)
        all_results[language] = results
        
    # Comparative analysis
    if len(all_results) == 2:
        print(f"\n{'='*60}")
        print("COMPARATIVE ANALYSIS")
        print(f"{'='*60}")
        
        jp_results = all_results.get("japanese", {})
        en_results = all_results.get("english", {})
        
        if jp_results and en_results:
            jp_basic = jp_results["basic_properties"]
            en_basic = en_results["basic_properties"]
            
            print("\n📊 Dimension Comparison:")
            print(f"  • Japanese: {jp_basic['vocab_size']:,} words × {jp_basic['embedding_dim']} dims")
            print(f"  • English: {en_basic['vocab_size']:,} words × {en_basic['embedding_dim']} dims")
            
            print("\n📈 Quality Comparison:")
            jp_assess = jp_results["assessment"]
            en_assess = en_results["assessment"]
            print(f"  • Japanese quality: {jp_assess['quality_score']}/10")
            print(f"  • English quality: {en_assess['quality_score']}/10")
            
            if jp_basic['embedding_dim'] == en_basic['embedding_dim']:
                print("\n✅ Embedding dimensions match - compatible for cross-lingual tasks")
            else:
                print("\n⚠️  Different embedding dimensions - may need alignment for cross-lingual use")
    
    return all_results


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    results = main()