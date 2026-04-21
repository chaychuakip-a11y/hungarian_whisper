"""
ROCm-compatible replacements for whisper_ED CUDA kernels.
Replaces custom CUDA operations with PyTorch native operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ScaledMaskedSoftmax(torch.autograd.Function):
    """ROCm-compatible scaled masked softmax.

    Replaces custom CUDA kernel with PyTorch native operations.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, mask: torch.Tensor, scale: float) -> torch.Tensor:
        input_scaled = input * scale
        if mask is not None:
            input_scaled = input_scaled.masked_fill(mask, float('-inf'))
        output = F.softmax(input_scaled, dim=-1)
        ctx.save_for_backward(output, mask, torch.tensor(scale))
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        output, mask, scale = ctx.saved_tensors
        grad_input = grad_output * output
        sum_grad = grad_input.sum(dim=-1, keepdim=True)
        grad_input = grad_input - output * sum_grad
        grad_input = grad_input * scale[0]
        return grad_input, None, None


def scaled_masked_softmax_forward(
    input: torch.Tensor,
    mask: Optional[torch.Tensor],
    scale: float
) -> torch.Tensor:
    """Forward pass of scaled masked softmax for ROCm."""
    return ScaledMaskedSoftmax.apply(input, mask, scale)


class FusedSoftmax(nn.Module):
    """Fused softmax module that replaces CUDA scaled_masked_softmax.

    Uses PyTorch native operations for ROCm compatibility.
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale

    def forward(self, input: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return scaled_masked_softmax_forward(input, mask, self.scale)


class ScaledUpperTriangMaskedSoftmax(torch.autograd.Function):
    """ROCm-compatible upper triangular masked softmax."""

    @staticmethod
    def forward(ctx, input: torch.Tensor, scale: float) -> torch.Tensor:
        seq_len = input.size(-1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=input.device, dtype=torch.bool), diagonal=1)
        input_scaled = input * scale
        input_scaled = input_scaled.masked_fill(mask, float('-inf'))
        output = F.softmax(input_scaled, dim=-1)
        ctx.save_for_backward(output, torch.tensor(scale))
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        output, scale = ctx.saved_tensors
        grad_input = grad_output * output
        sum_grad = grad_input.sum(dim=-1, keepdim=True)
        grad_input = grad_input - output * sum_grad
        grad_input = grad_input * scale[0]
        return grad_input, None


def scaled_upper_triang_masked_softmax_forward(
    input: torch.Tensor,
    scale: float
) -> torch.Tensor:
    """Forward pass for upper triangular masked softmax."""
    return ScaledUpperTriangMaskedSoftmax.apply(input, scale)


class LayerNorm(nn.Module):
    """ROCm-compatible LayerNorm replacement.

    Uses PyTorch's native LayerNorm for ROCm compatibility.
    """

    def __init__(self, normalized_shape, eps: float = 1e-05, elementwise_affine: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class RMSNorm(nn.Module):
    """ROCm-compatible RMSNorm replacement."""

    def __init__(self, normalized_shape, eps: float = 1e-05):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.weight is not None:
            output = output * self.weight
        return output
