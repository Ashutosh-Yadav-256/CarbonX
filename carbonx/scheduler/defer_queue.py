"""
Defer Queue Module

Manages deferred requests waiting for lower carbon intensity windows.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Any, Callable
from queue import PriorityQueue
import threading
import time
import structlog

logger = structlog.get_logger()


@dataclass(order=True)
class DeferredRequest:
    """A request waiting in the defer queue."""
    priority: float  # Lower = higher priority (execute first)
    request_id: str = field(compare=False)
    prompt: str = field(compare=False)
    max_tokens: int = field(compare=False, default=256)
    tenant_id: Optional[str] = field(compare=False, default=None)
    created_at: datetime = field(compare=False, default_factory=datetime.utcnow)
    execute_after: datetime = field(compare=False, default_factory=datetime.utcnow)
    max_wait_until: datetime = field(compare=False, default_factory=lambda: datetime.utcnow() + timedelta(hours=4))
    callback: Optional[Callable] = field(compare=False, default=None)
    
    @property
    def is_expired(self) -> bool:
        """Check if request has exceeded max wait time."""
        return datetime.utcnow() > self.max_wait_until
    
    @property
    def is_ready(self) -> bool:
        """Check if request is ready to execute."""
        return datetime.utcnow() >= self.execute_after


class DeferQueue:
    """
    Priority queue for deferred requests.
    
    Manages requests that are waiting for better carbon conditions
    and processes them when conditions improve or deadlines approach.
    """
    
    def __init__(
        self,
        max_queue_size: int = 1000,
        check_interval_seconds: float = 30.0,
    ):
        """
        Initialize the defer queue.
        
        Args:
            max_queue_size: Maximum requests to queue
            check_interval_seconds: How often to check for ready requests
        """
        self.max_queue_size = max_queue_size
        self.check_interval = check_interval_seconds
        
        self._queue: PriorityQueue = PriorityQueue()
        self._lock = threading.RLock()
        self._request_count = 0
        self._processed_count = 0
        self._expired_count = 0
        
        # Processing callback
        self._processor: Optional[Callable] = None
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        
        logger.info(
            "defer_queue_initialized",
            max_size=max_queue_size,
            check_interval=check_interval_seconds,
        )
    
    def enqueue(
        self,
        request_id: str,
        prompt: str,
        max_tokens: int = 256,
        tenant_id: Optional[str] = None,
        execute_after: Optional[datetime] = None,
        max_wait_hours: float = 4.0,
        priority: float = 0.5,
        callback: Optional[Callable] = None,
    ) -> bool:
        """
        Add a request to the defer queue.
        
        Args:
            request_id: Unique request identifier
            prompt: The inference prompt
            max_tokens: Maximum tokens to generate
            tenant_id: Optional tenant identifier
            execute_after: Earliest time to execute
            max_wait_hours: Maximum hours to wait
            priority: Priority (lower = higher priority)
            callback: Optional callback when processed
            
        Returns:
            True if enqueued successfully
        """
        with self._lock:
            if self._request_count >= self.max_queue_size:
                logger.warning("defer_queue_full", max_size=self.max_queue_size)
                return False
            
            request = DeferredRequest(
                priority=priority,
                request_id=request_id,
                prompt=prompt,
                max_tokens=max_tokens,
                tenant_id=tenant_id,
                execute_after=execute_after or datetime.utcnow(),
                max_wait_until=datetime.utcnow() + timedelta(hours=max_wait_hours),
                callback=callback,
            )
            
            self._queue.put(request)
            self._request_count += 1
            
            logger.info(
                "request_deferred",
                request_id=request_id,
                execute_after=request.execute_after.isoformat(),
                max_wait_until=request.max_wait_until.isoformat(),
            )
            
            return True
    
    def get_ready_requests(
        self,
        max_count: int = 10,
    ) -> list[DeferredRequest]:
        """
        Get requests that are ready to process.
        
        Args:
            max_count: Maximum requests to return
            
        Returns:
            List of ready DeferredRequest objects
        """
        ready = []
        requeue = []
        
        with self._lock:
            while not self._queue.empty() and len(ready) < max_count:
                request = self._queue.get_nowait()
                
                if request.is_expired:
                    self._expired_count += 1
                    logger.warning(
                        "request_expired",
                        request_id=request.request_id,
                    )
                    continue
                
                if request.is_ready:
                    ready.append(request)
                else:
                    requeue.append(request)
            
            # Put back requests that aren't ready yet
            for req in requeue:
                self._queue.put(req)
        
        return ready
    
    def set_processor(self, processor: Callable[[DeferredRequest], Any]) -> None:
        """
        Set the processor function for deferred requests.
        
        Args:
            processor: Function that takes a DeferredRequest and processes it
        """
        self._processor = processor
        logger.info("processor_set")
    
    def start_background_processing(self) -> None:
        """Start background thread to process ready requests."""
        if self._running:
            return
        
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._process_loop,
            daemon=True,
        )
        self._worker_thread.start()
        logger.info("background_processing_started")
    
    def stop_background_processing(self) -> None:
        """Stop background processing."""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)
        logger.info("background_processing_stopped")
    
    def _process_loop(self) -> None:
        """Background processing loop."""
        while self._running:
            try:
                ready = self.get_ready_requests()
                
                for request in ready:
                    if self._processor:
                        try:
                            result = self._processor(request)
                            self._processed_count += 1
                            
                            if request.callback:
                                request.callback(result)
                            
                            logger.info(
                                "deferred_request_processed",
                                request_id=request.request_id,
                                wait_time_s=(datetime.utcnow() - request.created_at).total_seconds(),
                            )
                        except Exception as e:
                            logger.error(
                                "deferred_request_failed",
                                request_id=request.request_id,
                                error=str(e),
                            )
                    
                    with self._lock:
                        self._request_count -= 1
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error("process_loop_error", error=str(e))
                time.sleep(self.check_interval)
    
    @property
    def queue_size(self) -> int:
        """Current queue size."""
        return self._request_count
    
    @property
    def stats(self) -> dict:
        """Queue statistics."""
        return {
            "queue_size": self._request_count,
            "processed_count": self._processed_count,
            "expired_count": self._expired_count,
            "max_size": self.max_queue_size,
        }
