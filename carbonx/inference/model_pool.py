"""
Model Pool Module

Manages the registry of model variants (Small, Medium, Large) with
their associated metadata and loading/inference capabilities.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any
import time
import threading
import structlog

logger = structlog.get_logger()


class ModelStatus(str, Enum):
    """Model loading status."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    name: str
    hf_model_id: str
    energy_per_token_kwh: float
    quality_score: float
    status: ModelStatus = ModelStatus.UNLOADED
    model: Any = None
    tokenizer: Any = None
    load_time_seconds: float = 0.0
    error_message: Optional[str] = None


@dataclass
class InferenceResult:
    """Result of model inference."""
    text: str
    tokens_generated: int
    input_tokens: int
    latency_ms: float
    model_name: str
    early_exit_layer: Optional[int] = None
    confidence: Optional[float] = None


class ModelPool:
    """
    Manages a pool of LLM model variants for adaptive inference.
    
    Supports:
    - Small: Fast, low-energy, lower quality
    - Medium: Balanced
    - Large: High quality, high energy
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize the model pool.
        
        Args:
            config: Optional configuration dict with model specs
        """
        self._models: dict[str, ModelInfo] = {}
        self._lock = threading.RLock()
        self._device = None
        self._torch_available = False
        
        # Check for torch availability
        try:
            import torch
            self._torch_available = True
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("torch_available", device=self._device)
        except ImportError:
            logger.warning("torch_not_available", fallback="mock_inference")
        
        # Register default models if no config provided
        if config is None:
            self._register_default_models()
        else:
            for name, spec in config.items():
                self.register_model(
                    name=name,
                    hf_model_id=spec["hf_model_id"],
                    energy_per_token_kwh=spec["energy_per_token_kwh"],
                    quality_score=spec["quality_score"],
                )
    
    def _register_default_models(self):
        """Register the default model variants."""
        models = [
            ("small", "distilgpt2", 1e-8, 0.7),
            ("medium", "gpt2", 3e-8, 0.85),
            ("large", "gpt2-large", 8e-8, 1.0),
        ]
        
        for name, hf_id, energy, quality in models:
            self.register_model(name, hf_id, energy, quality)
    
    def register_model(
        self,
        name: str,
        hf_model_id: str,
        energy_per_token_kwh: float,
        quality_score: float,
    ) -> None:
        """
        Register a model variant in the pool.
        
        Args:
            name: Model name (e.g., "small", "medium", "large")
            hf_model_id: HuggingFace model identifier
            energy_per_token_kwh: Energy consumption per token
            quality_score: Relative quality score (0-1)
        """
        with self._lock:
            self._models[name] = ModelInfo(
                name=name,
                hf_model_id=hf_model_id,
                energy_per_token_kwh=energy_per_token_kwh,
                quality_score=quality_score,
            )
            logger.info(
                "model_registered",
                name=name,
                hf_model_id=hf_model_id,
            )
    
    def load_model(self, name: str) -> bool:
        """
        Load a model into memory.
        
        Args:
            name: Model name to load
            
        Returns:
            True if loaded successfully
        """
        with self._lock:
            if name not in self._models:
                logger.error("model_not_found", name=name)
                return False
            
            info = self._models[name]
            
            if info.status == ModelStatus.READY:
                return True
            
            info.status = ModelStatus.LOADING
        
        start_time = time.time()
        
        try:
            if self._torch_available:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                
                logger.info("loading_model", name=name, hf_id=info.hf_model_id)
                
                tokenizer = AutoTokenizer.from_pretrained(info.hf_model_id)
                model = AutoModelForCausalLM.from_pretrained(info.hf_model_id)
                
                if self._device == "cuda":
                    model = model.cuda()
                
                # Set pad token if not set
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                with self._lock:
                    info.model = model
                    info.tokenizer = tokenizer
                    info.status = ModelStatus.READY
                    info.load_time_seconds = time.time() - start_time
                
                logger.info(
                    "model_loaded",
                    name=name,
                    load_time_s=info.load_time_seconds,
                    device=self._device,
                )
                return True
            else:
                # Mock model for environments without torch
                with self._lock:
                    info.model = "mock"
                    info.tokenizer = "mock"
                    info.status = ModelStatus.READY
                    info.load_time_seconds = time.time() - start_time
                
                logger.info("mock_model_loaded", name=name)
                return True
                
        except Exception as e:
            with self._lock:
                info.status = ModelStatus.ERROR
                info.error_message = str(e)
            logger.error("model_load_failed", name=name, error=str(e))
            return False
    
    def unload_model(self, name: str) -> None:
        """Unload a model from memory."""
        with self._lock:
            if name in self._models:
                info = self._models[name]
                info.model = None
                info.tokenizer = None
                info.status = ModelStatus.UNLOADED
                logger.info("model_unloaded", name=name)
    
    def get_model_info(self, name: str) -> Optional[ModelInfo]:
        """Get information about a model."""
        return self._models.get(name)
    
    def list_models(self) -> list[str]:
        """List all registered model names."""
        return list(self._models.keys())
    
    def get_available_models(self) -> list[str]:
        """List models that are ready for inference."""
        return [
            name for name, info in self._models.items()
            if info.status == ModelStatus.READY
        ]
    
    def inference(
        self,
        name: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> InferenceResult:
        """
        Run inference on a model.
        
        Args:
            name: Model name
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            InferenceResult with generated text and metadata
        """
        info = self._models.get(name)
        
        if info is None:
            raise ValueError(f"Model '{name}' not found in pool")
        
        if info.status != ModelStatus.READY:
            # Auto-load if not ready
            if not self.load_model(name):
                raise RuntimeError(f"Failed to load model '{name}'")
            info = self._models[name]
        
        start_time = time.time()
        
        if self._torch_available and info.model != "mock":
            import torch
            
            # Tokenize input
            inputs = info.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            )
            
            if self._device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            input_tokens = inputs["input_ids"].shape[1]
            
            # Generate
            with torch.no_grad():
                outputs = info.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=info.tokenizer.pad_token_id,
                )
            
            # Decode
            generated_tokens = outputs[0][input_tokens:]
            generated_text = info.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            )
            tokens_generated = len(generated_tokens)
            
        else:
            # Mock inference for testing
            input_tokens = len(prompt.split())
            tokens_generated = min(max_tokens, input_tokens * 2)
            generated_text = f"[Mock response from {name} model for: {prompt[:50]}...]"
        
        latency_ms = (time.time() - start_time) * 1000
        
        result = InferenceResult(
            text=generated_text,
            tokens_generated=tokens_generated,
            input_tokens=input_tokens,
            latency_ms=latency_ms,
            model_name=name,
        )
        
        logger.info(
            "inference_complete",
            model=name,
            tokens=tokens_generated,
            latency_ms=latency_ms,
        )
        
        return result
    
    def estimate_energy(self, name: str, num_tokens: int) -> float:
        """
        Estimate energy consumption for generating tokens.
        
        Args:
            name: Model name
            num_tokens: Number of tokens to generate
            
        Returns:
            Estimated energy in kWh
        """
        info = self._models.get(name)
        if info is None:
            return 0.0
        return info.energy_per_token_kwh * num_tokens
    
    def get_energy_rankings(self) -> list[tuple[str, float]]:
        """
        Get models ranked by energy efficiency (lowest first).
        
        Returns:
            List of (model_name, energy_per_token) tuples
        """
        rankings = [
            (name, info.energy_per_token_kwh)
            for name, info in self._models.items()
        ]
        return sorted(rankings, key=lambda x: x[1])
