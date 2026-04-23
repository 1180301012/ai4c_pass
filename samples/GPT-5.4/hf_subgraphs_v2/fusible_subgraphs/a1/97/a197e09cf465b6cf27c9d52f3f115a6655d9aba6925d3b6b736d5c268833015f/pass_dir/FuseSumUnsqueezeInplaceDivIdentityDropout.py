import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = in_0.sum(dim=-1)
    tmp_1 = tmp_0.unsqueeze(-1)
    tmp_2 = in_0 / tmp_1
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.0, False, False)
    return (tmp_3,)


def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    ],
    key=['N'],
)
@triton.jit
def _row_norm_kernel(
    x_ptr,
    out_ptr,
    num_rows,
    N,
    stride_row,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_rows:
        return

    row_start = x_ptr + pid * stride_row
    out_start = out_ptr + pid * stride_row

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(row_start + offs, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    denom = tl.sum(x_f32, axis=0)
    y = x_f32 / denom
    tl.store(out_start + offs, y.to(x.dtype), mask=mask)


@torch.fx.wrap
def fused_row_norm_identity_dropout(in_0):
    out = torch.empty_like(in_0)

    shape = in_0.shape
    N = shape[-1]

    num_rows = 1
    for i in range(in_0.dim() - 1):
        num_rows *= shape[i]

    stride_row = in_0.stride(-2)

    grid = (num_rows,)
    _row_norm_kernel[grid](
        in_0,
        out,
        num_rows,
        N,
        stride_row,
    )
    return (out,)


def replacement_func():
    return fused_row_norm_identity_dropout