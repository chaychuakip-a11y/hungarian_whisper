"""
ROCm compatibility layer for whisper_ED framework.

Usage:
    from hulk_rocm import patch_whisper_ed

    # Before importing whisper_ED modules
    patch_whisper_ed()

    # Now import and use whisper_ED normally
    from hulk import ...
"""

from .compat import (
    ScaledMaskedSoftmax,
    scaled_masked_softmax_forward,
    FusedSoftmax,
    ScaledUpperTriangMaskedSoftmax,
    scaled_upper_triang_masked_softmax_forward,
    LayerNorm,
    RMSNorm,
)
from .patch import (
    create_rocm_compat_softmax,
    patch_scaled_masked_softmax,
    HuggingFaceFlashAttention,
)


def patch_whisper_ed():
    """Patch whisper_ED for ROCm compatibility.

    This should be called before importing any whisper_ED modules.
    It replaces CUDA kernel references with ROCm-compatible alternatives.
    """
    import sys

    # Patch scaled masked softmax
    patch_scaled_masked_softmax()

    # Replace CUDA extensions with PyTorch native operations
    sys.modules['hulk.hulk.kernel.cuda_native.scaled_masked_softmax'] = __import__('hulk_rocm.compat')

    return True


__all__ = [
    'ScaledMaskedSoftmax',
    'scaled_masked_softmax_forward',
    'FusedSoftmax',
    'ScaledUpperTriangMaskedSoftmax',
    'scaled_upper_triang_masked_softmax_forward',
    'LayerNorm',
    'RMSNorm',
    'create_rocm_compat_softmax',
    'patch_scaled_masked_softmax',
    'HuggingFaceFlashAttention',
    'patch_whisper_ed',
]
