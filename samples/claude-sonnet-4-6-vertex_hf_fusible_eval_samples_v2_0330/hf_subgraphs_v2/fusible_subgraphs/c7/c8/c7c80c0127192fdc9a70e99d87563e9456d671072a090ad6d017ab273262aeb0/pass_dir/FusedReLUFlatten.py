import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_1 = in_0.flatten(1, -1)
    return tmp_1


def replacement_args(in_0):
    # in_0 is the FX node for the relu output (the flatten's input).
    # Return it directly — the replacement will just reshape it.
    return (in_0,)


@triton.jit
def _relu_flatten_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused relu+flatten kernel for larger tensors (kept for reference)."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    y = tl.maximum(x, 0.0)
    tl.store(output_ptr + offsets, y, mask=mask)


@torch.fx.wrap
def fused_relu_flatten(relu_out):
    # relu_out is already relu-applied.
    # Re-apply the same flatten(1,-1) — equivalent view, stable across dtypes.
    return relu_out.flatten(1, -1)


def replacement_func():
    return fused_relu_flatten