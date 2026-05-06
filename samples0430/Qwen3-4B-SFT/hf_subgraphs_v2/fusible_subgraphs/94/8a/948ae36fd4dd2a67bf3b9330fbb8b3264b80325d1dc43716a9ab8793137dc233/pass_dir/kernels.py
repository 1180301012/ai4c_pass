"""
Shared Triton kernels and dispatch wrapper for int64→bool cast operations.
All FuseArangeBoolCast_N* pass files import dispatch_bool_cast from here
so that replacement_func() returns the SAME function object in every pass
(file), preventing output_pass_replacement_func_limit from dropping passes.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _cast_i64_to_bool_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Casts int64 input (flat) to bool: non-zero -> True, zero -> False."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask, other=1)
    out = (x != 0).to(tl.int1)
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def dispatch_bool_cast(in_0, route):
    """
    Generic bool-cast replacement for in_0.to(device='cuda', dtype=torch.bool).
    `route` is injected by replacement_args to allow the dispatch wrapper to be
    shared across all FuseArangeBoolCast_* passes (satisfying replacement_func_limit).

    Performance:
    - out = empty(n_elements, bool, 'cuda')  → avoids dtype-inheritance pitfall
    - flat-write via Triton, then .view() (= zero-copy reshape, same as native .reshape)
    - BLOCK_SIZE chosen so grid has ≤32 blocks for the small-tensor case
    """
    n_elements = in_0.numel()
    if n_elements <= 32:
        BLOCK_SIZE = 32
    elif n_elements <= 512:
        BLOCK_SIZE = 64
    elif n_elements <= 8192:
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = 1024

    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    out = torch.empty(n_elements, dtype=torch.bool, device='cuda')
    _cast_i64_to_bool_kernel[grid](in_0, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    # view is a zero-copy meta-operation (no CUDA kernel), same as .reshape()
    return out.view(*in_0.shape)