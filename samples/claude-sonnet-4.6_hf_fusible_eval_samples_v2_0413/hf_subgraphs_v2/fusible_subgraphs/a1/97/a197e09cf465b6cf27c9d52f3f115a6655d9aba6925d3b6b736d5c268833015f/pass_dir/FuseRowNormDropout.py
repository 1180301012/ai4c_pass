import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = in_0.sum(dim=-1)
    tmp_1 = tmp_0.unsqueeze(-1)
    in_0 /= tmp_1
    tmp_3 = torch.nn.functional.dropout(in_0, 0.0, False, False)
    return tmp_3


def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['n_cols'],
)
@triton.jit
def _row_norm_kernel(
    x_ptr,
    out_ptr,
    n_cols,
    row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    row_start = row_id * row_stride
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    # Load one full row (masked beyond n_cols)
    x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)

    # Accumulate sum in float32 for numerical stability across dtypes
    x_f32 = x.to(tl.float32)
    row_sum = tl.sum(x_f32, axis=0)

    # Normalise: divide by row sum, cast back to original dtype
    out = (x_f32 / row_sum).to(x.dtype)

    # Store result
    tl.store(out_ptr + row_start + offsets, out, mask=mask)


@torch.fx.wrap
def row_norm_dropout_wrapper(in_0):
    n_cols = in_0.shape[-1]
    n_rows = in_0.numel() // n_cols
    # stride of the last-but-one dimension gives the row stride in elements
    row_stride = in_0.stride(-2) if in_0.dim() >= 2 else n_cols

    out = torch.empty_like(in_0)

    _row_norm_kernel[(n_rows,)](
        x_ptr=in_0,
        out_ptr=out,
        n_cols=n_cols,
        row_stride=row_stride,
    )

    return out


def replacement_func():
    return row_norm_dropout_wrapper