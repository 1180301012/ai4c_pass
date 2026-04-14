"""
Shared Triton kernel for bool-cast optimization.
Imported by OptimizeBoolCast pass file.

NOTE: torch.arange is intentionally NOT handled here because it has no tensor
inputs and would be constant-folded by FX (producing a get_attr/_tensor_constant0
node) rather than a call_function node, making pattern matching impossible.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _bool_cast_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Cast int64 tensor elements to bool (stored as int8 / uint8)."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask, other=0)
    out = (x != 0).to(tl.int8)
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def optimized_bool_cast(in_0):
    """
    Triton-accelerated replacement for:
        tmp_2 = in_0.to(device=cuda, dtype=torch.bool)

    Uses a fixed BLOCK_SIZE=1024 (no autotune) to avoid per-call
    config-sweep overhead inside torch.compile.
    """
    n_elements = in_0.numel()
    bool_out = torch.empty(in_0.shape, dtype=torch.bool, device=in_0.device)
    BLOCK_SIZE = 1024
    _bool_cast_kernel[
        (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,
    ](in_0, bool_out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return bool_out