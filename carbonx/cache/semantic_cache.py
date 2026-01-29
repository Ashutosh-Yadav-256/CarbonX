"""
Semantic Cache Module

Implements response caching based on semantic similarity to avoid
redundant inference for similar queries.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
import hashlib
import threading
import structlog

logger = structlog.get_logger()


@dataclass
class CacheEntry:
    """A cached response entry."""
    prompt_hash: str
    prompt: str
    response: str
    model_used: str
    carbon_gco2: float
    created_at: datetime
    access_count: int = 0
    last_accessed: datetime = None
    
    def __post_init__(self):
        if self.last_accessed is None:
            self.last_accessed = self.created_at


@dataclass
class CacheHit:
    """Result of a cache lookup."""
    hit: bool
    entry: Optional[CacheEntry] = None
    similarity: float = 0.0


class SemanticCache:
    """
    Caches LLM responses based on semantic similarity.
    
    For similar prompts, returns cached responses to avoid
    redundant computation and carbon emissions.
    
    Uses:
    - Exact hash matching for identical prompts
    - Embedding-based similarity for semantic matching (optional)
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.92,
        max_size: int = 10000,
        ttl_hours: float = 24.0,
        use_embeddings: bool = False,
    ):
        """
        Initialize the semantic cache.
        
        Args:
            similarity_threshold: Minimum similarity for cache hit
            max_size: Maximum number of entries
            ttl_hours: Time-to-live for entries
            use_embeddings: Whether to use embedding-based matching
        """
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self.ttl_hours = ttl_hours
        self.use_embeddings = use_embeddings
        
        # Storage
        self._cache: dict[str, CacheEntry] = {}
        self._embeddings: dict[str, list[float]] = {}
        self._lock = threading.RLock()
        
        # Stats
        self._hits = 0
        self._misses = 0
        self._carbon_saved_gco2 = 0.0
        
        # Embedding model (lazy loaded)
        self._embedding_model = None
        
        logger.info(
            "semantic_cache_initialized",
            max_size=max_size,
            ttl_hours=ttl_hours,
            use_embeddings=use_embeddings,
        )
    
    def _get_hash(self, prompt: str) -> str:
        """Get deterministic hash of a prompt."""
        normalized = prompt.strip().lower()
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]
    
    def _load_embedding_model(self) -> bool:
        """Lazy load the embedding model."""
        if self._embedding_model is not None:
            return True
        
        if not self.use_embeddings:
            return False
        
        try:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("embedding_model_loaded")
            return True
        except ImportError:
            logger.warning("sentence_transformers_not_available")
            self.use_embeddings = False
            return False
        except Exception as e:
            logger.warning("embedding_model_load_failed", error=str(e))
            self.use_embeddings = False
            return False
    
    def _get_embedding(self, text: str) -> Optional[list[float]]:
        """Get embedding for text."""
        if not self._load_embedding_model():
            return None
        
        try:
            embedding = self._embedding_model.encode(text, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as e:
            logger.debug("embedding_failed", error=str(e))
            return None
    
    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between embeddings."""
        dot = sum(x * y for x, y in zip(a, b))
        return dot  # Already normalized
    
    def lookup(self, prompt: str) -> CacheHit:
        """
        Look up a prompt in the cache.
        
        Args:
            prompt: The query prompt
            
        Returns:
            CacheHit with result
        """
        prompt_hash = self._get_hash(prompt)
        
        with self._lock:
            # Try exact match first
            if prompt_hash in self._cache:
                entry = self._cache[prompt_hash]
                
                # Check TTL
                age = datetime.utcnow() - entry.created_at
                if age.total_seconds() / 3600 > self.ttl_hours:
                    del self._cache[prompt_hash]
                    if prompt_hash in self._embeddings:
                        del self._embeddings[prompt_hash]
                else:
                    entry.access_count += 1
                    entry.last_accessed = datetime.utcnow()
                    self._hits += 1
                    self._carbon_saved_gco2 += entry.carbon_gco2
                    
                    logger.info(
                        "cache_hit",
                        prompt_hash=prompt_hash,
                        carbon_saved=entry.carbon_gco2,
                    )
                    
                    return CacheHit(hit=True, entry=entry, similarity=1.0)
            
            # Try semantic matching if enabled
            if self.use_embeddings:
                query_embedding = self._get_embedding(prompt)
                if query_embedding:
                    best_match = None
                    best_similarity = 0.0
                    
                    for hash_key, embedding in self._embeddings.items():
                        similarity = self._cosine_similarity(query_embedding, embedding)
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = hash_key
                    
                    if best_match and best_similarity >= self.similarity_threshold:
                        entry = self._cache.get(best_match)
                        if entry:
                            entry.access_count += 1
                            entry.last_accessed = datetime.utcnow()
                            self._hits += 1
                            self._carbon_saved_gco2 += entry.carbon_gco2
                            
                            logger.info(
                                "semantic_cache_hit",
                                similarity=best_similarity,
                                carbon_saved=entry.carbon_gco2,
                            )
                            
                            return CacheHit(
                                hit=True,
                                entry=entry,
                                similarity=best_similarity,
                            )
        
        self._misses += 1
        return CacheHit(hit=False)
    
    def store(
        self,
        prompt: str,
        response: str,
        model_used: str,
        carbon_gco2: float,
    ) -> None:
        """
        Store a response in the cache.
        
        Args:
            prompt: The query prompt
            response: The model response
            model_used: Which model was used
            carbon_gco2: Carbon emitted for this response
        """
        prompt_hash = self._get_hash(prompt)
        
        with self._lock:
            # Evict if at capacity
            if len(self._cache) >= self.max_size:
                self._evict_lru()
            
            entry = CacheEntry(
                prompt_hash=prompt_hash,
                prompt=prompt,
                response=response,
                model_used=model_used,
                carbon_gco2=carbon_gco2,
                created_at=datetime.utcnow(),
            )
            
            self._cache[prompt_hash] = entry
            
            # Store embedding if enabled
            if self.use_embeddings:
                embedding = self._get_embedding(prompt)
                if embedding:
                    self._embeddings[prompt_hash] = embedding
        
        logger.debug(
            "cache_stored",
            prompt_hash=prompt_hash,
            model=model_used,
            carbon=carbon_gco2,
        )
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return
        
        oldest_hash = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_accessed,
        )
        
        del self._cache[oldest_hash]
        if oldest_hash in self._embeddings:
            del self._embeddings[oldest_hash]
        
        logger.debug("cache_evicted", prompt_hash=oldest_hash)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._embeddings.clear()
        logger.info("cache_cleared")
    
    @property
    def stats(self) -> dict:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "carbon_saved_gco2": self._carbon_saved_gco2,
        }
