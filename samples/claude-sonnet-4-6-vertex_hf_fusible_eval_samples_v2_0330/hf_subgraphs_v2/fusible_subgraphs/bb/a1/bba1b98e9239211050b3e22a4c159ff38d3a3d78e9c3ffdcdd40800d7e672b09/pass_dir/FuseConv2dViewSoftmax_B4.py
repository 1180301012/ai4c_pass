"""
Fused pass: conv2d (1x1, Cout=1) + view(4,1,-1) + softmax(dim=-1)
Batch size B=4, spatial HW=4096, channels C=512.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 256, 'BLOCK_C': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 256, 'BLOCK_C': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 512, 'BLOCK_C': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 512, 'BLOCK_C': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 1024,'BLOCK_C': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 1024,'BLOCK_C': 128}, num_warps=8, num_stages=2),
    ],
    key=['C', 'HW'],
)
@triton.jit
def _dotprod_kernel_b4(
    x_ptr, w_ptr, b_ptr, y_ptr,
    B, C, HW,
    BLOCK_HW: tl.constexpr,
    BLOCK_C:  tl.constexpr,
):
    pid_b  = tl.program_id(0)
    pid_hw = tl.program_id(1)

    hw_base = pid_hw * BLOCK_HW
    hw_offs = hw_base + tl.arange(0, BLOCK_HW)

    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)

    for c_base in range(0, C, BLOCK_C):
        c_offs = c_base + tl.arange(0, BLOCK_C)
        w = tl.load(w_ptr + c_offs).to(tl.float32)
        x = tl.load(
            x_ptr + pid_b * C * HW + c_offs[:, None] * HW + hw_offs[None, :]
        ).to(tl.float32)
        acc += tl.sum(x * w[:, None], axis=0)

    bias = tl.load(b_ptr).to(tl.float32)
    acc += bias
    tl.store(y_ptr + pid_b * HW + hw_offs, acc)


@triton.jit
def _softmax_kernel_b4(
    out_ptr, inp_ptr,
    HW: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = tl.arange(0, HW)
    x    = tl.load(inp_ptr + pid * HW + offs)
    m    = tl.max(x, axis=0)
    e    = tl.exp(x - m)
    s    = tl.sum(e, axis=0)
    tl.store(out_ptr + pid * HW + offs, (e / s).to(out_ptr.dtype.element_ty))


@torch.fx.wrap
def fused_conv_softmax_b4(in_0, in_1, in_2):
    B, C, H, W = in_2.shape
    HW = H * W  # 4096

    mid = torch.empty((B, HW), device=in_2.device, dtype=torch.float32)

    grid_conv = lambda meta: (B, triton.cdiv(HW, meta['BLOCK_HW']))
    _dotprod_kernel_b4[grid_conv](
        in_2, in_1.reshape(-1), in_0,
        mid,
        B, C, HW,
    )

    out = torch.empty((B, HW), device=in_2.device, dtype=in_2.dtype)
    _softmax_kernel_b4[(B,)](
        out, mid,
        HW=4096,
        num_warps=8,
    )

    return out.view(B, 1, HW)


def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3  = conv2d.view(4, 1, -1)
    tmp_4  = tmp_3.softmax(dim=-1)
    return (tmp_4,)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_conv_softmax_b4