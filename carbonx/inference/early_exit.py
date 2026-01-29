"""
Early Exit Module

Implements token-level early exit inference to reduce computation
when confidence threshold is met.

Formula: Exit at layer k if Conf(k) ≥ θ
"""

from dataclasses import dataclass
from typing import Optional, Any, Callable
import time
import structlog

logger = structlog.get_logger()


@dataclass
class EarlyExitResult:
    """Result of early-exit inference."""
    text: str
    tokens_generated: int
    exit_layer: Optional[int] = None
    total_layers: int = 0
    average_confidence: float = 0.0
    computation_saved_ratio: float = 0.0
    latency_ms: float = 0.0


class EarlyExitWrapper:
    """
    Wraps model inference with early-exit capability.
    
    Monitors confidence at intermediate layers and terminates
    generation early when threshold is satisfied.
    
    Note: True early-exit requires model architecture modifications.
    This implementation provides a simulation-based approach that
    demonstrates the concept and energy savings calculation.
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.85,
        min_layers: int = 2,
        enabled: bool = True,
    ):
        """
        Initialize early exit wrapper.
        
        Args:
            confidence_threshold: Minimum confidence to trigger exit (θ)
            min_layers: Minimum layers to process before considering exit
            enabled: Whether early exit is active
        """
        self.confidence_threshold = confidence_threshold
        self.min_layers = min_layers
        self.enabled = enabled
        
        logger.info(
            "early_exit_initialized",
            threshold=confidence_threshold,
            min_layers=min_layers,
            enabled=enabled,
        )
    
    def should_exit(self, confidence: float, current_layer: int) -> bool:
        """
        Determine if inference should exit early.
        
        Args:
            confidence: Current prediction confidence
            current_layer: Current layer number (0-indexed)
            
        Returns:
            True if should exit early
        """
        if not self.enabled:
            return False
        
        if current_layer < self.min_layers:
            return False
        
        return confidence >= self.confidence_threshold
    
    def compute_savings(
        self,
        exit_layer: int,
        total_layers: int,
    ) -> float:
        """
        Compute the ratio of computation saved by early exit.
        
        Args:
            exit_layer: Layer at which exit occurred
            total_layers: Total layers in the model
            
        Returns:
            Ratio of computation saved (0-1)
        """
        if total_layers <= 0:
            return 0.0
        
        layers_skipped = total_layers - exit_layer - 1
        return max(0.0, layers_skipped / total_layers)
    
    def wrap_inference(
        self,
        inference_fn: Callable,
        prompt: str,
        max_tokens: int = 256,
        **kwargs,
    ) -> EarlyExitResult:
        """
        Wrap a standard inference function with early-exit logic.
        
        This is a simulation-based implementation that:
        1. Runs standard inference
        2. Simulates early-exit based on token confidence
        3. Calculates potential energy savings
        
        For production use, this should be integrated with a model
        architecture that supports intermediate layer outputs.
        
        Args:
            inference_fn: The base inference function to wrap
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments for inference_fn
            
        Returns:
            EarlyExitResult with exit metadata
        """
        start_time = time.time()
        
        # Run base inference
        result = inference_fn(prompt=prompt, max_tokens=max_tokens, **kwargs)
        
        # Simulate early exit analysis
        # In production, this would access intermediate layer outputs
        exit_layer = None
        total_layers = 12  # Typical for small models like GPT-2
        computation_saved = 0.0
        
        if self.enabled:
            # Simulate confidence-based early exit
            # Shorter, simpler responses typically have higher confidence
            simulated_confidence = self._estimate_confidence(
                result.text if hasattr(result, 'text') else str(result),
                result.tokens_generated if hasattr(result, 'tokens_generated') else max_tokens,
            )
            
            if simulated_confidence >= self.confidence_threshold:
                # Simulate exiting at earlier layer
                # Higher confidence = earlier exit
                exit_ratio = 1.0 - (simulated_confidence - self.confidence_threshold) / 0.15
                exit_layer = max(
                    self.min_layers,
                    int(total_layers * exit_ratio)
                )
                computation_saved = self.compute_savings(exit_layer, total_layers)
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Extract text and tokens from result
        if hasattr(result, 'text'):
            text = result.text
            tokens = result.tokens_generated
        else:
            text = str(result)
            tokens = len(text.split())
        
        early_result = EarlyExitResult(
            text=text,
            tokens_generated=tokens,
            exit_layer=exit_layer,
            total_layers=total_layers,
            average_confidence=self._estimate_confidence(text, tokens),
            computation_saved_ratio=computation_saved,
            latency_ms=latency_ms,
        )
        
        if exit_layer is not None:
            logger.info(
                "early_exit_triggered",
                exit_layer=exit_layer,
                total_layers=total_layers,
                saved_ratio=computation_saved,
            )
        
        return early_result
    
    def _estimate_confidence(self, text: str, tokens: int) -> float:
        """
        Estimate confidence for simulated early exit.
        
        Uses heuristics based on response characteristics.
        In production, this would use actual model confidence scores.
        """
        # Shorter responses often have higher confidence (less uncertainty)
        length_factor = max(0.5, 1.0 - (tokens / 500))
        
        # Responses with hedging language have lower confidence
        hedging_words = ["maybe", "perhaps", "might", "could", "possibly", "uncertain"]
        hedging_count = sum(1 for word in hedging_words if word in text.lower())
        hedging_factor = max(0.6, 1.0 - (hedging_count * 0.1))
        
        # Combine factors
        confidence = 0.7 * length_factor + 0.3 * hedging_factor
        
        return min(0.99, max(0.5, confidence))
    
    def adjust_energy_for_early_exit(
        self,
        base_energy_kwh: float,
        computation_saved_ratio: float,
    ) -> float:
        """
        Adjust energy consumption based on early exit savings.
        
        Args:
            base_energy_kwh: Original energy estimate
            computation_saved_ratio: Ratio of computation saved
            
        Returns:
            Adjusted energy in kWh
        """
        return base_energy_kwh * (1.0 - computation_saved_ratio)
