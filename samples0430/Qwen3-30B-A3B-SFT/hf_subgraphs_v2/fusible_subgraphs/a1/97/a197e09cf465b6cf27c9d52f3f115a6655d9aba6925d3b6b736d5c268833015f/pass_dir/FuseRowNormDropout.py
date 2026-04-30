import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern – sum + unsqueeze.
# This is the largest sub-pattern confirmed to match.
# The replacement computes sum(row) for each row; Inductor handles the
# remaining unsqueeze (view) + div after the kernel.
# ---------------------------------------------------------------------------
def pattern(in_0):
    tmp_0 = in_0.sum(dim=-1)
    tmp_1 = tmp_0.unsqueeze(-1)
    return tmp_1


def replacement_args(in_0):
    return (in_0,)


# ---------------------------------------------------------------------------
# Triton kernel – fast row-sum (fp32 accumulation for precision)
# One program per row.  BLOCK_N must be >= N (next power-of-2 ≥ 196 → 256)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 512}, num_warps=8, num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def _row_norm_kernel(
    x_ptr,
    out_ptr,
    N,
    stride_row,
    BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_offset = row_idx * stride_row

    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N

    row = tl.load(x_ptr + row_offset + offsets, mask=mask, other=0.0)
    row_sum = tl.sum(row, axis=0)   # scalar sum of the row

    # Store one scalar per row into the [num_rows] output
    tl.store(out_ptr + row_idx, row_sum)


# ---------------------------------------------------------------------------
# Wrapper – called instead of the matched sum + unsqueeze
# Returns [B*S] flat array; the unsqueeze is handled by the caller/view
# ---------------------------------------------------------------------------
@torch.fx.wrap
def _row_norm_wrapper(in_0):
    N = in_0.shape[-1]
    num_rows = in_0.numel() // N

    # Output: [B, S, 1] to match tmp_0.unsqueeze(-1) output shape
    out = torch.empty(in_0.shape[:-1] + (1,), dtype=in_0.dtype, device=in_0.device)

    _row_norm_kernel[(num_rows,)](
        in_0,
        out,
        N,
        N,   # stride_row = N (contiguous row-major)
    )

    return out


def replacement_func():
    return _row_norm_wrapper