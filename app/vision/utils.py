"""
Shared utility functions for vision processing modules.
"""

def get_memory_usage_mb() -> float:
    """Get current memory usage in MB.
    
    Returns:
        Memory usage in MB, or 0.0 if psutil not available
    """
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0