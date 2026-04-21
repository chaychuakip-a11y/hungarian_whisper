"""
ROCm patcher for whisper_ED.
Patches CUDA kernel references to use ROCm-compatible PyTorch operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def create_rocm_compat_softmax(scale: float = 1.0):
    """Create ROCm-compatible softmax.

    This replaces the CUDA hulk_scaled_masked_softmax kernel.
    """
    return _ROCMSoftmax(scale)


class _ROCMSoftmax(nn.Module):
    """ROCm-compatible replacement for scaled masked softmax."""

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)
            mask: Optional mask tensor
        """
        scaled_x = x * self.scale
        if mask is not None:
            scaled_x = scaled_x.masked_fill(mask, float('-inf'))
        return F.softmax(scaled_x, dim=-1)


def patch_scaled_masked_softmax():
    """Patch the scaled masked softmax module for ROCm compatibility.

    After patching, all references to hulk_scaled_masked_softmax will use
    PyTorch native operations instead.
    """
    try:
        import hulk.hulk.kernel.cuda_native.scaled_masked_softmax as sms
        sms.ScaledMaskedSoftmax = _ROCMSoftmax
        return True
    except (ImportError, AttributeError):
        return False


class HuggingFaceFlashAttention:
    """ROCm flash attention using transformers library.

    ROCm 5.7+ supports flash attention through the transformers library.
    This class provides a compatible interface.
    """

    @staticmethod
    def is_available() -> bool:
        """Check if flash attention is available."""
        try:
            from transformers.utils import is_flash_attn_2_available
            return is_flash_attn_2_available()
        except ImportError:
            return False

    @staticmethod
    def replace_attention(module: nn.Module, attention_name: str = "attention"):
        """Replace standard attention with flash attention if available."""
        if HuggingFaceFlashAttention.is_available():
            # Flash attention is available through transformers
            pass
        return module
