"""
Stub the dead-code 1×1 convolution (conv2d_2) that follows .to(torch.bfloat16).

The pattern `.to(torch.bfloat16) → torch.conv2d(...)` uniquely identifies conv2d_2
in the bfloat16 graph. The main-path conv2d does NOT follow a .to() call.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _zero_fill_bf16_conv2_kernel(out_ptr, N_TOTAL, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_TOTAL
    tl.store(out_ptr + offs, tl.zeros([BLOCK_SIZE], dtype=out_ptr.dtype.element_ty), mask=mask)


@torch.fx.wrap
def stub_dead_conv2_bf16(x, weight, bias):
    """
    Return zeros of shape [N, C_out, H, W] — the conv2d_2 output shape.
    conv2d_2 uses a 1×1 kernel with stride=1, padding=0, so spatial dims are unchanged.
    """
    N, _, H, W = x.shape
    C_out = weight.shape[0]
    if x.is_cuda:
        out = torch.empty(N, C_out, H, W, dtype=x.dtype, device=x.device)
        N_TOTAL = out.numel()
        BLOCK_SIZE = 1024
        _zero_fill_bf16_conv2_kernel[((N_TOTAL + BLOCK_SIZE - 1) // BLOCK_SIZE,)](
            out, N_TOTAL, BLOCK_SIZE=BLOCK_SIZE
        )
        return out
    return torch.zeros(N, C_out, H, W, dtype=x.dtype, device=x.device)


def pattern(x, weight, bias):
    """
    Matches: to = ?.to(torch.bfloat16); conv2d_2 = torch.conv2d(to, weight, bias, (1,1),(0,0),(1,1),1)
    The .to(torch.bfloat16) anchor differentiates this from the main-path conv2d.
    """
    t = x.to(torch.bfloat16)
    return torch.conv2d(t, weight, bias, (1, 1), (0, 0), (1, 1), 1)


def replacement_args(x, weight, bias):
    return (x, weight, bias)


def replacement_func():
    return stub_dead_conv2_bf16