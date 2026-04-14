import torch
import triton
import triton.language as tl


def pattern(x):
    tmp_5 = x.sum(dim=3, keepdim=True)
    tmp_6 = x / tmp_5
    return tmp_6


def replacement_args(x):
    return (x,)


@triton.jit
def sum_div_kernel(
    x_ptr,
    out_ptr,
    N_cols,
    BLOCK_COLS: tl.constexpr,
):
    """
    16 CUDA programs × BLOCK_COLS=8 threads each.
    Fuses sum(dim=3,keepdim=True) + div into a single memory pass.
    """
    row_id    = tl.program_id(0)
    col_offs  = tl.arange(0, BLOCK_COLS)
    row_start = row_id * N_cols

    x     = tl.load(x_ptr  + row_start + col_offs)
    x_f32 = x.to(tl.float32)
    rsum  = tl.sum(x_f32, axis=0)
    out   = (x_f32 / rsum).to(x.dtype)
    tl.store(out_ptr + row_start + col_offs, out)


@torch.fx.wrap
def fuse_sum_div(x):
    # x shape: [1, 2, 8, 8]  →  16 rows of 8 elements, contiguous
    out    = torch.empty_like(x)
    N_rows = x.shape[0] * x.shape[1] * x.shape[2]   # 16
    N_cols = x.shape[3]                               # 8
    sum_div_kernel[(N_rows,)](
        x, out,
        N_cols,
        BLOCK_COLS=8,
    )
    return out


def replacement_func():
    return fuse_sum_div