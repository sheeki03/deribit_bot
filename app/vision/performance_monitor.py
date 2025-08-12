"""
Performance monitoring and metrics collection for vision processing.

This module provides utilities to track performance metrics, rate limiting,
and resource usage during image analysis operations.
"""
from __future__ import annotations

import time
import json
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import contextmanager
from collections import deque, defaultdict

logger = logging.getLogger(__name__)

@dataclass 
class ProcessingMetrics:
    """Metrics for a single processing operation."""
    operation: str
    start_time: float
    end_time: float
    duration: float
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return self.duration * 1000

@dataclass
class SessionMetrics:
    """Aggregated metrics for a processing session."""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    total_operations: int
    successful_operations: int
    failed_operations: int
    total_duration: float
    avg_duration: float
    operations_per_second: float
    peak_memory_mb: float
    error_rate: float
    operations: List[ProcessingMetrics]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            **asdict(self),
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'operations': [asdict(op) for op in self.operations]
        }

class RateLimiter:
    """Token bucket rate limiter for API calls."""
    
    def __init__(self, rate: float, burst: int = 1):
        """
        Args:
            rate: Operations per second
            burst: Maximum burst size (number of tokens in bucket)
        """
        self.rate = rate
        self.burst = burst
        self.tokens = burst
        self.last_update = time.time()
    
    def acquire(self, timeout: float = 10.0) -> bool:
        """Acquire a token, blocking if necessary.
        
        Args:
            timeout: Maximum time to wait for a token
            
        Returns:
            True if token acquired, False if timeout
        """
        start_time = time.time()
        
        while True:
            now = time.time()
            elapsed = now - self.last_update
            
            # Add tokens based on elapsed time
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            
            if now - start_time >= timeout:
                return False
            
            # Sleep for a short time to avoid busy waiting
            time.sleep(min(0.1, (1 - self.tokens) / self.rate))
    
    def try_acquire(self) -> bool:
        """Try to acquire a token without blocking."""
        now = time.time()
        elapsed = now - self.last_update
        
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
        self.last_update = now
        
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False

class PerformanceMonitor:
    """Monitor and track performance metrics for vision processing."""
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or f"session_{int(time.time())}"
        self.start_time = datetime.now()
        self.operations: List[ProcessingMetrics] = []
        self.peak_memory_mb = 0.0
        self.operation_counts: Dict[str, int] = defaultdict(int)
        self.operation_durations: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        from .utils import get_memory_usage_mb
        return get_memory_usage_mb()
    
    @contextmanager
    def track_operation(self, operation: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager to track an operation."""
        start_time = time.time()
        success = False
        error = None
        
        try:
            yield
            success = True
        except Exception as e:
            error = str(e)
            raise
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            # Track memory usage
            current_memory = self._get_memory_usage_mb()
            self.peak_memory_mb = max(self.peak_memory_mb, current_memory)
            
            # Create metrics record
            metrics = ProcessingMetrics(
                operation=operation,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                success=success,
                error=error,
                metadata=metadata
            )
            
            self.operations.append(metrics)
            self.operation_counts[operation] += 1
            self.operation_durations[operation].append(duration)
            
            # Log slow operations
            if duration > 10.0:  # More than 10 seconds
                logger.warning(
                    f"Slow operation detected: {operation} took {duration:.2f}s",
                    extra={'operation': operation, 'duration': duration, 'metadata': metadata}
                )
    
    def get_operation_stats(self, operation: str) -> Dict[str, Any]:
        """Get statistics for a specific operation type."""
        durations = list(self.operation_durations[operation])
        if not durations:
            return {'count': 0, 'avg_duration': 0.0, 'min_duration': 0.0, 'max_duration': 0.0}
        
        return {
            'count': len(durations),
            'avg_duration': sum(durations) / len(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'p95_duration': sorted(durations)[int(0.95 * len(durations))] if len(durations) >= 20 else max(durations)
        }
    
    def get_session_metrics(self) -> SessionMetrics:
        """Get comprehensive session metrics."""
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        successful_ops = sum(1 for op in self.operations if op.success)
        failed_ops = len(self.operations) - successful_ops
        
        avg_duration = sum(op.duration for op in self.operations) / len(self.operations) if self.operations else 0.0
        ops_per_second = len(self.operations) / total_duration if total_duration > 0 else 0.0
        error_rate = failed_ops / len(self.operations) if self.operations else 0.0
        
        return SessionMetrics(
            session_id=self.session_id,
            start_time=self.start_time,
            end_time=end_time,
            total_operations=len(self.operations),
            successful_operations=successful_ops,
            failed_operations=failed_ops,
            total_duration=total_duration,
            avg_duration=avg_duration,
            operations_per_second=ops_per_second,
            peak_memory_mb=self.peak_memory_mb,
            error_rate=error_rate,
            operations=self.operations
        )
    
    def save_metrics(self, output_path: Path) -> None:
        """Save metrics to a JSON file."""
        metrics = self.get_session_metrics()
        
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(metrics.to_dict(), f, indent=2)
            
            logger.info(f"Performance metrics saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    def log_summary(self) -> None:
        """Log a summary of performance metrics."""
        metrics = self.get_session_metrics()
        
        logger.info("=== PERFORMANCE SUMMARY ===")
        logger.info(f"Session: {metrics.session_id}")
        logger.info(f"Duration: {metrics.total_duration:.1f}s")
        logger.info(f"Operations: {metrics.total_operations} ({metrics.successful_operations} success, {metrics.failed_operations} failed)")
        logger.info(f"Throughput: {metrics.operations_per_second:.2f} ops/sec")
        logger.info(f"Average operation time: {metrics.avg_duration:.3f}s")
        logger.info(f"Error rate: {metrics.error_rate:.1%}")
        logger.info(f"Peak memory usage: {metrics.peak_memory_mb:.1f}MB")
        
        # Operation type breakdown
        if self.operation_counts:
            logger.info("Operation breakdown:")
            for op_type, count in self.operation_counts.items():
                stats = self.get_operation_stats(op_type)
                logger.info(f"  {op_type}: {count} ops, avg {stats['avg_duration']:.3f}s")

class AdaptiveRateLimiter:
    """Rate limiter that adapts based on error rates."""
    
    def __init__(self, initial_rate: float = 1.0, min_rate: float = 0.1, max_rate: float = 10.0):
        self.rate_limiter = RateLimiter(initial_rate)
        self.initial_rate = initial_rate
        self.current_rate = initial_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        
        # Adaptive parameters
        self.recent_errors = deque(maxlen=20)
        self.recent_successes = deque(maxlen=20)
        self.last_adjustment = time.time()
        self.adjustment_interval = 30.0  # Adjust every 30 seconds
    
    def record_result(self, success: bool, error_type: Optional[str] = None) -> None:
        """Record the result of an operation."""
        if success:
            self.recent_successes.append(time.time())
        else:
            self.recent_errors.append((time.time(), error_type))
        
        self._maybe_adjust_rate()
    
    def _maybe_adjust_rate(self) -> None:
        """Adjust rate based on recent error patterns."""
        now = time.time()
        if now - self.last_adjustment < self.adjustment_interval:
            return
        
        self.last_adjustment = now
        
        # Calculate recent error rate
        recent_operations = len(self.recent_errors) + len(self.recent_successes)
        if recent_operations < 5:  # Need some data
            return
        
        error_rate = len(self.recent_errors) / recent_operations
        
        # Check for rate limiting errors
        rate_limit_errors = sum(1 for _, error_type in self.recent_errors 
                               if error_type and '429' in str(error_type))
        
        if rate_limit_errors > 0:
            # Reduce rate due to rate limiting
            new_rate = max(self.min_rate, self.current_rate * 0.5)
            logger.info(f"Rate limit detected, reducing rate from {self.current_rate:.2f} to {new_rate:.2f}")
        elif error_rate < 0.05:  # Less than 5% error rate
            # Increase rate
            new_rate = min(self.max_rate, self.current_rate * 1.2)
            if new_rate != self.current_rate:
                logger.info(f"Low error rate, increasing rate from {self.current_rate:.2f} to {new_rate:.2f}")
        elif error_rate > 0.2:  # More than 20% error rate
            # Decrease rate
            new_rate = max(self.min_rate, self.current_rate * 0.8)
            logger.info(f"High error rate ({error_rate:.1%}), reducing rate from {self.current_rate:.2f} to {new_rate:.2f}")
        else:
            return  # No adjustment needed
        
        self.current_rate = new_rate
        self.rate_limiter = RateLimiter(new_rate)
    
    def acquire(self, timeout: float = 10.0) -> bool:
        """Acquire a token with adaptive rate limiting."""
        return self.rate_limiter.acquire(timeout)