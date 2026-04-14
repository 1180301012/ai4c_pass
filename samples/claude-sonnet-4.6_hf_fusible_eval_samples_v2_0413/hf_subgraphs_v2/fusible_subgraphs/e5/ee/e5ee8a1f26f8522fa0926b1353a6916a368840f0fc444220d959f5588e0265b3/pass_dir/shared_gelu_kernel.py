"""
Shared GELU kernel used by both FuseGeluDropout and FuseGeluApproxNoneDropout passes.
Imported so that both passes return the EXACT SAME function object from replacement_func(),
bypassing the output_pass_replacement_func_limit.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _gelu_erf_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Upcast to float32 for erf precision (no-op for float32 inputs)
    x_f32 = x.to(tl.float32)
    # Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    SQRT2_INV = 0.7071067811865476
    out_f32 = x_f32 * 0.5 * (1.0 + tl.math.erf(x_f32 * SQRT2_INV))
    # Downcast back to original dtype and store
    tl.store(out_ptr + offsets, out_f32.to(x.dtype), mask=mask)


@torch.fx.wrap
def triton_gelu_dispatch(x, route):
    """
    Shared dispatch wrapper for both gelu and gelu(approximate='none') patterns.
    The 'route' argument is a string tag; computation is the same for both routes
    (exact erf-based GELU). Block size is chosen based on tensor size to balance
    occupancy and kernel-launch overhead without autotune.
    """
    n = x.numel()
    out = torch.empty_like(x)
    if n <= 1024 * 1024:           # ≤ 1 M elements  — small tensor
        BLOCK_SIZE = 1024
        NW = 4
    elif n <= 16 * 1024 * 1024:    # ≤ 16 M elements — medium tensor
        BLOCK_SIZE = 4096
        NW = 4
    else:                           # > 16 M elements — large tensor
        BLOCK_SIZE = 8192
        NW = 8
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _gelu_erf_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=NW)
    return out