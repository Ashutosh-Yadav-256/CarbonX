"""
Workload Generator Module

Generates synthetic workloads for simulation and testing.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import random
import structlog

logger = structlog.get_logger()


class QueryType(str, Enum):
    """Types of queries for simulation."""
    FACTUAL = "factual"
    REASONING = "reasoning"
    CODE = "code"
    CREATIVE = "creative"


@dataclass
class SyntheticRequest:
    """A synthetic inference request."""
    request_id: str
    prompt: str
    query_type: QueryType
    expected_complexity: str  # low, medium, high
    max_tokens: int
    is_urgent: bool


# Sample prompts by type
SAMPLE_PROMPTS = {
    QueryType.FACTUAL: [
        "What is the capital of France?",
        "Who invented the telephone?",
        "What year did World War II end?",
        "What is the chemical formula for water?",
        "How many continents are there?",
        "What is the speed of light?",
        "Who wrote Romeo and Juliet?",
        "What is the largest planet in our solar system?",
    ],
    QueryType.REASONING: [
        "Explain how photosynthesis works and why it's important for life on Earth.",
        "Compare and contrast renewable and non-renewable energy sources.",
        "Analyze the impact of climate change on global agriculture.",
        "Discuss the ethical implications of artificial intelligence in healthcare.",
        "Explain the causes and effects of inflation in modern economies.",
        "How does machine learning differ from traditional programming?",
        "What are the pros and cons of electric vehicles?",
        "Explain how vaccines work to protect against diseases.",
    ],
    QueryType.CODE: [
        "Write a Python function to calculate the Fibonacci sequence.",
        "Implement a binary search algorithm in JavaScript.",
        "Create a SQL query to find duplicate records in a table.",
        "Write a recursive function to reverse a linked list.",
        "Implement a simple REST API endpoint using Flask.",
        "Create a function to validate email addresses using regex.",
        "Write code to parse JSON and extract specific fields.",
        "Implement a basic sorting algorithm and explain its complexity.",
    ],
    QueryType.CREATIVE: [
        "Write a short poem about the beauty of nature.",
        "Create a story opening about a mysterious traveler.",
        "Describe a futuristic city in vivid detail.",
        "Write dialogue for two characters meeting for the first time.",
        "Create a product description for an innovative gadget.",
        "Write a motivational message for someone starting a new job.",
        "Describe a perfect sunset scene.",
        "Create a backstory for a fantasy character.",
    ],
}


class WorkloadGenerator:
    """
    Generates synthetic workloads for simulation.
    
    Supports customizable:
    - Request volume
    - Complexity distribution
    - Query type mix
    - Urgency patterns
    """
    
    def __init__(
        self,
        complexity_distribution: Optional[dict[str, float]] = None,
        query_type_distribution: Optional[dict[str, float]] = None,
        urgency_ratio: float = 0.7,
        seed: Optional[int] = None,
    ):
        """
        Initialize the workload generator.
        
        Args:
            complexity_distribution: Distribution of low/medium/high complexity
            query_type_distribution: Distribution of query types
            urgency_ratio: Ratio of urgent requests (0-1)
            seed: Random seed for reproducibility
        """
        self.complexity_dist = complexity_distribution or {
            "low": 0.4,
            "medium": 0.4,
            "high": 0.2,
        }
        
        self.query_type_dist = query_type_distribution or {
            "factual": 0.35,
            "reasoning": 0.35,
            "code": 0.2,
            "creative": 0.1,
        }
        
        self.urgency_ratio = urgency_ratio
        
        if seed is not None:
            random.seed(seed)
        
        logger.info(
            "workload_generator_initialized",
            complexity_dist=self.complexity_dist,
            query_type_dist=self.query_type_dist,
        )
    
    def generate_request(self, request_id: Optional[str] = None) -> SyntheticRequest:
        """Generate a single synthetic request."""
        # Select query type
        query_type = self._weighted_choice(
            list(QueryType),
            [self.query_type_dist.get(t.value, 0.25) for t in QueryType],
        )
        
        # Select prompt
        prompts = SAMPLE_PROMPTS.get(query_type, SAMPLE_PROMPTS[QueryType.FACTUAL])
        prompt = random.choice(prompts)
        
        # Determine complexity based on query type
        if query_type == QueryType.FACTUAL:
            complexity = "low"
            max_tokens = random.randint(50, 150)
        elif query_type in [QueryType.REASONING, QueryType.CODE]:
            complexity = random.choices(
                ["medium", "high"],
                weights=[0.6, 0.4],
            )[0]
            max_tokens = random.randint(200, 500)
        else:
            complexity = "medium"
            max_tokens = random.randint(150, 300)
        
        # Override with configured distribution
        complexity = self._weighted_choice(
            ["low", "medium", "high"],
            [self.complexity_dist.get(c, 0.33) for c in ["low", "medium", "high"]],
        )
        
        # Urgency
        is_urgent = random.random() < self.urgency_ratio
        
        return SyntheticRequest(
            request_id=request_id or f"req_{random.randint(10000, 99999)}",
            prompt=prompt,
            query_type=query_type,
            expected_complexity=complexity,
            max_tokens=max_tokens,
            is_urgent=is_urgent,
        )
    
    def generate_batch(
        self,
        count: int,
        id_prefix: str = "sim",
    ) -> list[SyntheticRequest]:
        """
        Generate a batch of synthetic requests.
        
        Args:
            count: Number of requests to generate
            id_prefix: Prefix for request IDs
            
        Returns:
            List of SyntheticRequest objects
        """
        requests = []
        for i in range(count):
            request_id = f"{id_prefix}_{i:05d}"
            requests.append(self.generate_request(request_id))
        
        logger.info("batch_generated", count=count)
        return requests
    
    def _weighted_choice(self, choices: list, weights: list):
        """Select from choices with weights."""
        return random.choices(choices, weights=weights)[0]
    
    def get_workload_stats(self, requests: list[SyntheticRequest]) -> dict:
        """Get statistics about a generated workload."""
        if not requests:
            return {}
        
        complexity_counts = {"low": 0, "medium": 0, "high": 0}
        type_counts = {t.value: 0 for t in QueryType}
        urgent_count = 0
        
        for req in requests:
            complexity_counts[req.expected_complexity] += 1
            type_counts[req.query_type.value] += 1
            if req.is_urgent:
                urgent_count += 1
        
        total = len(requests)
        
        return {
            "total_requests": total,
            "complexity_distribution": {k: v / total for k, v in complexity_counts.items()},
            "query_type_distribution": {k: v / total for k, v in type_counts.items()},
            "urgency_ratio": urgent_count / total,
            "avg_max_tokens": sum(r.max_tokens for r in requests) / total,
        }
