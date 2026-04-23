import torch
import triton
import triton.language as tl
from graph_net_bench.torch.posion_dispatch_tensor import unwrap_tensor


def pattern(in_1, in_2, in_3):
    tmp_4 = torch.nn.functional.layer_norm(in_3, (768,), in_2, in_1, 1e-12)
    return tmp_4


def replacement_args(in_1, in_2, in_3):
    return (in_1, in_2, in_3)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
    ],
    key=["N_COLS"],
)
@triton.jit
def layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    N_ROWS,
    N_COLS,
    EPS,
    IS_BF16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N_COLS
    row_start = row * N_COLS

    x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N_COLS
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / N_COLS
    rstd = tl.rsqrt(var + EPS)

    w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x_centered * rstd * w + b

    if IS_BF16:
        y = y.to(tl.bfloat16)
    else:
        y = y.to(tl.float16)

    tl.store(out_ptr + row_start + cols, y, mask=mask)


@torch.fx.wrap
def triton_layer_norm_inner(in_1, in_2, in_3):
    in_1 = unwrap_tensor(in_1)
    in_2 = unwrap_tensor(in_2)
    in_3 = unwrap_tensor(in_3)

    hidden = in_3.shape[-1]
    rows = in_3.numel() // hidden
    out = torch.empty_like(in_3)

    layer_norm_kernel[(rows,)](
        in_3,
        in_2,
        in_1,
        out,
        rows,
        hidden,
        1e-12,
        in_3.dtype == torch.bfloat16,
        BLOCK_SIZE=1024,
    )
    return out


def triton_layer_norm(in_1, in_2, in_3):
    return triton_layer_norm_inner(in_1, in_2, in_3)


def replacement_func():
    return triton_layer_norm