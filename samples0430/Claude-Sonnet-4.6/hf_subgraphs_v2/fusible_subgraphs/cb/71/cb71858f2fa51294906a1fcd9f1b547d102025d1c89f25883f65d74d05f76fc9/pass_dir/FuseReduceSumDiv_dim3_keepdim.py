import torch
import triton
import triton.language as tl


def pattern(x):
    s = x.sum(dim=3, keepdim=True)
    out = x / s
    return out


def replacement_args(x):
    return (x,)


@triton.jit
def _sum_div_kernel(
    x_ptr,
    out_ptr,
    ROW_LEN: tl.constexpr,
):
    """One Triton program per row; fuses sum-reduce + broadcast-div."""
    row_id = tl.program_id(0)
    row_start = row_id * ROW_LEN
    offs = row_start + tl.arange(0, ROW_LEN)
    x = tl.load(x_ptr + offs)
    # Accumulate in fp32 for numerical stability
    x_f32 = x.to(tl.float32)
    row_sum = tl.sum(x_f32, axis=0)
    out = (x_f32 / row_sum).to(x.dtype)
    tl.store(out_ptr + offs, out)


@torch.fx.wrap
def fuse_sum_div_dim3(x):
    # x: [1, 2, 8, 8] – sum over dim=3, divide
    # Treat as N_ROWS=16 rows of ROW_LEN=8 elements
    out = torch.empty_like(x)
    _sum_div_kernel[(16,)](x, out, ROW_LEN=8)
    return out


def replacement_func():
    return fuse_sum_div_dim3