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
        triton.Config({'BLOCK_SIZE': 256}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
    ],
    key=['n_cols'],
)
@triton.jit
def l1_normalize_kernel(
    input_ptr,
    output_ptr,
    n_cols,
    input_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    row_start = row_idx * input_row_stride
    x = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=0.0)

    # Compute sum for normalization
    row_sum = tl.sum(x, axis=0)

    # Normalize
    out = x / row_sum

    # Store
    tl.store(output_ptr + row_start + col_offsets, out, mask=mask)


@torch.fx.wrap
def l1_normalize(in_0):
    shape = in_0.shape
    n_cols = shape[-1]
    n_rows = in_0.numel() // n_cols

    out = torch.empty_like(in_0)

    l1_normalize_kernel[(n_rows,)](
        input_ptr=in_0,
        output_ptr=out,
        n_cols=n_cols,
        input_row_stride=n_cols,
    )

    return out


def replacement_func():
    return l1_normalize