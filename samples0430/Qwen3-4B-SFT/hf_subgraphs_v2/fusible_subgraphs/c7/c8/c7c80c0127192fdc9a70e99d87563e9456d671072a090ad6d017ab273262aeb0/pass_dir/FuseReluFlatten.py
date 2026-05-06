import torch
import triton
import triton.language as tl


def pattern(in_0):
    return in_0.flatten(1, -1)


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def flat_copy_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(out_ptr + offsets, tl.load(x_ptr + offsets, mask=mask), mask=mask)


@torch.fx.wrap
def flatten_copy(in_0):
    # Input shape [B, C, 1, 1] (or any contiguous [B, C, 1, 1]).
    # After flatten dims 1..-1 the shape is [B, C].
    B, C = in_0.shape[0], in_0.numel() // in_0.shape[0]
    out = torch.empty(B, C, dtype=in_0.dtype, device=in_0.device)
    # Coalesced flat copy via Triton — no relu, just reshape via copy
    flat_copy_kernel[
        (triton.cdiv(B * C, 8192),)
    ](in_0, out, B * C, BLOCK_SIZE=8192)
    return out


def replacement_func():
    return flatten_copy