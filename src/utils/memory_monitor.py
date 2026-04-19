"""VRAM memory monitoring utilities."""

import gc
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """Monitor GPU memory usage during training.

    Helps prevent OOM errors by tracking VRAM allocation.

    Args:
        max_vram_gb: Maximum VRAM in GB (triggers warnings above this).
        device: CUDA device to monitor.
    """

    def __init__(self, max_vram_gb: float = 24.0, device: Optional[torch.device] = None):
        self.max_vram_gb = max_vram_gb
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.peak_memory_gb = 0.0

    def get_memory_stats(self) -> dict:
        """Get current memory statistics.

        Returns:
            Dict with allocated, reserved, and peak memory in GB.
        """
        if not torch.cuda.is_available():
            return {"allocated": 0, "reserved": 0, "peak": 0}

        stats = {
            "allocated": torch.cuda.memory_allocated(self.device) / 1e9,
            "reserved": torch.cuda.memory_reserved(self.device) / 1e9,
            "peak": torch.cuda.max_memory_allocated(self.device) / 1e9
        }
        self.peak_memory_gb = max(self.peak_memory_gb, stats["allocated"])

        return stats

    def log_memory_stats(self, prefix: str = "") -> None:
        """Log current memory statistics.

        Args:
            prefix: Prefix string for log message.
        """
        stats = self.get_memory_stats()
        msg = (
            f"{prefix} | "
            f"Allocated: {stats['allocated']:.2f}GB | "
            f"Reserved: {stats['reserved']:.2f}GB | "
            f"Peak: {stats['peak']:.2f}GB"
        )
        logger.info(msg)

    def check_oom(self) -> bool:
        """Check if memory usage is critically high.

        Returns:
            True if allocated memory > 90% of max.
        """
        if not torch.cuda.is_available():
            return False

        allocated = torch.cuda.memory_allocated(self.device) / 1e9
        return allocated > (self.max_vram_gb * 0.9)

    def get_peak_memory(self) -> float:
        """Get peak memory usage in GB.

        Returns:
            Peak allocated memory in GB.
        """
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated(self.device) / 1e9

    def reset_peak_stats(self) -> None:
        """Reset peak memory statistics."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
        self.peak_memory_gb = 0.0

    def clear_cache(self) -> None:
        """Clear GPU cache and run garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def print_memory_summary() -> None:
    """Print a summary of CUDA memory usage."""
    if not torch.cuda.is_available():
        logger.info("CUDA not available")
        return

    logger.info("=" * 60)
    logger.info("CUDA Memory Summary")
    logger.info("=" * 60)
    logger.info(f"Device: {torch.cuda.get_device_name(0)}")
    logger.info(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")
    logger.info(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
    logger.info(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f}GB")
    logger.info(f"Peak Allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
    logger.info("=" * 60)


def estimate_required_memory(
    model_params: int,
    batch_size: int,
    sequence_length: int,
    dtype_size_bytes: int = 2
) -> float:
    """Estimate memory required for training.

    Rough estimation for planning purposes.

    Args:
        model_params: Number of model parameters.
        batch_size: Training batch size.
        sequence_length: Max sequence length.
        dtype_size_bytes: Size of dtype in bytes (2 for bf16/fp16).

    Returns:
        Estimated memory in GB.
    """
    # Model weights (in bytes)
    model_bytes = model_params * dtype_size_bytes

    # Activations (rough estimate: 4 * batch * seq_len * hidden_size)
    # This is very rough
    activation_bytes = batch_size * sequence_length * 4096 * 4 * dtype_size_bytes

    # Gradients (same as model weights)
    gradient_bytes = model_bytes

    # Optimizer states (if Adam, 2x model size for first/second moment)
    optimizer_bytes = model_params * dtype_size_bytes * 2

    total_bytes = model_bytes + activation_bytes + gradient_bytes + optimizer_bytes
    return total_bytes / 1e9
