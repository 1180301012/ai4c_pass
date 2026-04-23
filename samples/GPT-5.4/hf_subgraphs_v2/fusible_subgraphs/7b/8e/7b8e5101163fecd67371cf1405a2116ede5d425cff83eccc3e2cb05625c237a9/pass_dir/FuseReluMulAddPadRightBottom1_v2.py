import operator
import torch
import triton
import triton.language as tl
from graph_net_bench.torch import custom_replacement as _cr

_cr.force_args_symbolic_trace = torch.fx.symbolic_trace


def pattern(in_0, in_1, in_2):
    tmp_2 = torch.nn.functional.relu(in_2, inplace=False)
    tmp_3 = operator.mul(in_1, tmp_2)
    tmp_4 = operator.add(tmp_3, in_0)
    tmp_5 = torch._C._nn.pad(tmp_4, (0, 1, 0, 1), 'constant', None)
    return tmp_5


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_W': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 256}, num_warps=8, num_stages=2),
    ],
    key=['W_OUT'],
)
@triton.jit
def fused_relu_mul_add_pad_rb1_kernel(
    bias_ptr,
    scale_ptr,
    x_ptr,
    out_ptr,
    H,
    W,
    H_OUT,
    W_OUT,
    BLOCK_W: tl.constexpr,
):
    pid_w = tl.program_id(0)
    pid_row = tl.program_id(1)

    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    h_out = pid_row % H_OUT
    nc = pid_row // H_OUT

    safe_h = tl.minimum(h_out, H - 1)
    in_row = nc * H + safe_h

    mask_out = offs_w < W_OUT
    mask_in = (h_out < H) & (offs_w < W)

    x = tl.load(x_ptr + in_row * W + offs_w, mask=mask_in, other=0)
    zero = tl.zeros((BLOCK_W,), dtype=x.dtype)
    x = tl.maximum(x, zero)

    scale = tl.load(scale_ptr)
    bias = tl.load(bias_ptr)
    y = x * scale + bias
    out = tl.where(mask_in, y, zero)

    tl.store(out_ptr + pid_row * W_OUT + offs_w, out, mask=mask_out)


@torch.fx.wrap
def fused_relu_mul_add_pad_rb1(bias, scale, x):
    n, c, h, w = x.shape
    h_out = h + 1
    w_out = w + 1
    out = torch.empty((n, c, h_out, w_out), device=x.device, dtype=x.dtype)

    grid = (triton.cdiv(w_out, 256), n * c * h_out)
    fused_relu_mul_add_pad_rb1_kernel[grid](
        bias,
        scale,
        x,
        out,
        h,
        w,
        h_out,
        w_out,
    )
    return out


def replacement_func():
    return fused_relu_mul_add_pad_rb1