import torch
import triton
import triton.language as tl


def pattern(in_0):
    # Matches: max(x, keepdim=True)[0].expand_as(x) - x
    # Anchor = operator.sub. Softmax that follows remains in graph.
    tmp_0 = torch.max(in_0, -1, keepdim=True)
    tmp_1 = tmp_0[0]
    tmp_2 = tmp_1.expand_as(in_0)
    tmp_3 = tmp_2 - in_0
    return tmp_3


def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 512}, num_warps=4),
        triton.Config({'BLOCK_N': 512}, num_warps=8),
        triton.Config({'BLOCK_N': 512}, num_warps=16),
    ],
    key=['n_cols'],
)
@triton.jit
def _max_shift_kernel(
    in_ptr,
    out_ptr,
    n_cols,
    BLOCK_N: tl.constexpr,
):
    """Computes max(x, dim=-1, keepdim=True).expand_as(x) - x row-wise."""
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < n_cols

    x = tl.load(in_ptr + row_idx * n_cols + offsets, mask=mask, other=-float('inf'))
    x_f32 = x.to(tl.float32)

    max_val = tl.max(x_f32, axis=0)
    result_f32 = max_val - x_f32

    result = result_f32.to(x.dtype)
    tl.store(out_ptr + row_idx * n_cols + offsets, result, mask=mask)


@torch.fx.wrap
def max_shift_wrapper(in_0):
    n_cols = in_0.shape[-1]
    n_rows = in_0.numel() // n_cols
    out = torch.empty_like(in_0)
    _max_shift_kernel[(n_rows,)](in_0, out, n_cols)
    return out


def replacement_func():
    return max_shift_wrapper