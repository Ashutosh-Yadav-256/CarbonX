"""
Complexity Estimator Module

Predicts computational complexity of inference requests to enable
adaptive model selection.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import re
import structlog

logger = structlog.get_logger()


class ComplexityLevel(str, Enum):
    """Request complexity levels."""
    LOW = "low"          # Simple factual queries
    MEDIUM = "medium"    # Standard queries
    HIGH = "high"        # Complex reasoning tasks


@dataclass
class ComplexityEstimate:
    """Result of complexity estimation."""
    level: ComplexityLevel
    confidence: float
    estimated_tokens: int
    features: dict
    
    def to_dict(self) -> dict:
        return {
            "level": self.level.value,
            "confidence": self.confidence,
            "estimated_tokens": self.estimated_tokens,
            "features": self.features,
        }


class ComplexityEstimator:
    """
    Estimates the computational complexity of inference requests.
    
    Uses lightweight heuristics based on:
    - Token count
    - Query structure (questions, reasoning keywords)
    - Historical patterns
    """
    
    # Keywords indicating simple factual queries
    FACTUAL_KEYWORDS = {
        "what is", "who is", "when did", "where is", "define",
        "list", "name", "how many", "what are",
    }
    
    # Keywords indicating complex reasoning
    REASONING_KEYWORDS = {
        "explain", "why", "how does", "analyze", "compare",
        "contrast", "evaluate", "discuss", "argue", "prove",
        "derive", "solve", "calculate", "design", "implement",
        "what if", "suppose", "assume", "consider",
    }
    
    # Code-related keywords
    CODE_KEYWORDS = {
        "code", "function", "class", "implement", "algorithm",
        "program", "debug", "fix", "refactor", "optimize",
        "python", "javascript", "java", "c++", "sql",
    }
    
    def __init__(
        self,
        low_token_threshold: int = 50,
        high_token_threshold: int = 200,
        base_output_ratio: float = 2.0,
    ):
        """
        Initialize the complexity estimator.
        
        Args:
            low_token_threshold: Max input tokens for LOW complexity
            high_token_threshold: Min input tokens for HIGH complexity
            base_output_ratio: Expected output/input token ratio
        """
        self.low_token_threshold = low_token_threshold
        self.high_token_threshold = high_token_threshold
        self.base_output_ratio = base_output_ratio
    
    def estimate(self, prompt: str, max_tokens: int = 256) -> ComplexityEstimate:
        """
        Estimate the complexity of a prompt.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum output tokens requested
            
        Returns:
            ComplexityEstimate with level, confidence, and features
        """
        # Extract features
        features = self._extract_features(prompt)
        
        # Score calculation
        score = self._calculate_score(features, max_tokens)
        
        # Determine level
        if score < 0.35:
            level = ComplexityLevel.LOW
        elif score < 0.65:
            level = ComplexityLevel.MEDIUM
        else:
            level = ComplexityLevel.HIGH
        
        # Estimate output tokens
        estimated_tokens = self._estimate_output_tokens(
            features, max_tokens, level
        )
        
        # Confidence based on feature clarity
        confidence = self._calculate_confidence(features, score)
        
        estimate = ComplexityEstimate(
            level=level,
            confidence=confidence,
            estimated_tokens=estimated_tokens,
            features=features,
        )
        
        logger.debug(
            "complexity_estimated",
            level=level.value,
            score=score,
            confidence=confidence,
            features=features,
        )
        
        return estimate
    
    def _extract_features(self, prompt: str) -> dict:
        """Extract features from the prompt."""
        prompt_lower = prompt.lower()
        words = prompt_lower.split()
        
        # Token count (rough estimate: 1 word ≈ 1.3 tokens)
        word_count = len(words)
        estimated_input_tokens = int(word_count * 1.3)
        
        # Keyword matching
        has_factual = any(kw in prompt_lower for kw in self.FACTUAL_KEYWORDS)
        has_reasoning = any(kw in prompt_lower for kw in self.REASONING_KEYWORDS)
        has_code = any(kw in prompt_lower for kw in self.CODE_KEYWORDS)
        
        # Structural features
        question_count = prompt.count("?")
        sentence_count = len(re.split(r'[.!?]+', prompt))
        has_list = bool(re.search(r'\d+\.|[-*•]', prompt))
        has_code_block = "```" in prompt or "def " in prompt or "function " in prompt
        
        return {
            "word_count": word_count,
            "estimated_input_tokens": estimated_input_tokens,
            "has_factual_keywords": has_factual,
            "has_reasoning_keywords": has_reasoning,
            "has_code_keywords": has_code,
            "question_count": question_count,
            "sentence_count": sentence_count,
            "has_list": has_list,
            "has_code_block": has_code_block,
        }
    
    def _calculate_score(self, features: dict, max_tokens: int) -> float:
        """Calculate complexity score (0-1)."""
        score = 0.0
        
        # Token-based scoring
        input_tokens = features["estimated_input_tokens"]
        if input_tokens < self.low_token_threshold:
            score += 0.1
        elif input_tokens > self.high_token_threshold:
            score += 0.3
        else:
            score += 0.2
        
        # Requested output length
        if max_tokens > 500:
            score += 0.2
        elif max_tokens > 200:
            score += 0.1
        
        # Keyword-based scoring
        if features["has_factual_keywords"] and not features["has_reasoning_keywords"]:
            score -= 0.15
        if features["has_reasoning_keywords"]:
            score += 0.25
        if features["has_code_keywords"] or features["has_code_block"]:
            score += 0.2
        
        # Structural complexity
        if features["question_count"] > 2:
            score += 0.1
        if features["has_list"]:
            score += 0.05
        
        return max(0.0, min(1.0, score))
    
    def _estimate_output_tokens(
        self,
        features: dict,
        max_tokens: int,
        level: ComplexityLevel,
    ) -> int:
        """Estimate expected output tokens."""
        input_tokens = features["estimated_input_tokens"]
        
        # Adjust ratio based on complexity
        if level == ComplexityLevel.LOW:
            ratio = self.base_output_ratio * 0.5
        elif level == ComplexityLevel.MEDIUM:
            ratio = self.base_output_ratio
        else:
            ratio = self.base_output_ratio * 1.5
        
        estimated = int(input_tokens * ratio)
        
        # Adjust for code (typically more verbose)
        if features["has_code_keywords"] or features["has_code_block"]:
            estimated = int(estimated * 1.5)
        
        # Cap at max_tokens
        return min(estimated, max_tokens)
    
    def _calculate_confidence(self, features: dict, score: float) -> float:
        """Calculate confidence in the complexity estimate."""
        # Higher confidence when features clearly indicate complexity
        confidence = 0.6  # Base confidence
        
        # Clear factual query
        if features["has_factual_keywords"] and not features["has_reasoning_keywords"]:
            confidence += 0.2
        
        # Clear reasoning query
        if features["has_reasoning_keywords"]:
            confidence += 0.15
        
        # Code queries are usually well-defined
        if features["has_code_keywords"]:
            confidence += 0.1
        
        # Extreme scores are more confident
        if score < 0.2 or score > 0.8:
            confidence += 0.1
        
        return min(0.95, confidence)
