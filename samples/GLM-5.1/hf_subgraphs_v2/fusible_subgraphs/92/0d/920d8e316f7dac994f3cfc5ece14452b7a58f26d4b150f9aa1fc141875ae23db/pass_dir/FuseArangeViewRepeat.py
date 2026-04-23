import torch
import triton
import triton.language as tl
from torch import device


@triton.jit
def arange_view_repeat_kernel(
    out_ptr,
    N: tl.constexpr,
    REPEAT_ROWS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Directly generate a (REPEAT_ROWS, N) tensor where each row contains [0, 1, ..., N-1].
    This fuses arange(N) + view(1, N) + repeat(REPEAT_ROWS, 1) into one kernel.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Each element's value is just its column index (row doesn't matter, all rows are identical)
    vals = offsets.to(tl.int64) % N
    # Total elements = REPEAT_ROWS * N
    mask = offsets < REPEAT_ROWS * N
    tl.store(out_ptr + offsets, vals, mask=mask)


@torch.fx.wrap
def fused_arange_view_repeat_dispatch(end, repeat_rows, dtype, route):
    """Dispatch wrapper that handles all route variants."""
    N = end
    BLOCK_SIZE = 1024
    total_elements = repeat_rows * N
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty((repeat_rows, N), dtype=dtype, device='cuda')

    arange_view_repeat_kernel[(num_programs,)](
        out_ptr=out,
        N=N,
        REPEAT_ROWS=repeat_rows,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def pattern():
    """Match the pattern: arange(0, 128) → view(1, -1) → repeat(2, 1)"""
    tmp_0 = torch.arange(0, 128, device=device(type='cuda'))
    tmp_1 = tmp_0.view(1, -1)
    tmp_2 = tmp_1.repeat(2, 1)
    return tmp_2


def replacement_args():
    """Return constants needed for the replacement, plus route string."""
    return (128, 2, torch.int64, "route_128")


def replacement_func():
    return fused_arange_view_repeat_dispatch