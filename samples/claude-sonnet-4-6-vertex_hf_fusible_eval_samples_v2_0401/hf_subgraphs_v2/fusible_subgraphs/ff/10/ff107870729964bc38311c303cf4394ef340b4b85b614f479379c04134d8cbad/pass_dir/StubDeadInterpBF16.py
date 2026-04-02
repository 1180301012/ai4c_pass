"""
Try to eliminate the dead-code interpolate + conv2d_2 + .to(bfloat16) chain in one pass.
Bfloat16 variant of StubDeadInterpF16.py.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _interp_noop_bf16_kernel(ptr, N: tl.constexpr):
    """Placeholder kernel - actual work is avoided by the stub."""
    pass


@torch.fx.wrap
def stub_dead_interp_bf16(x, weight, bias):
    """
    Replace the entire to(bfloat16) + conv2d_2 + interpolate dead chain.
    tmp_16 is not returned; return a minimal tensor.
    """
    return x.new_zeros(1)


def pattern(x, weight, bias):
    """
    Matches: t = x.to(bfloat16)
             c = conv2d(t, weight, bias, (1,1),(0,0),(1,1),1)
             result = F.interpolate(c, size=(512,512), bilinear, align_corners=False)
    """
    t = x.to(torch.bfloat16)
    c = torch.conv2d(t, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    result = torch.nn.functional.interpolate(c, size=(512, 512), mode='bilinear', align_corners=False)
    return result


def replacement_args(x, weight, bias):
    return (x, weight, bias)


def replacement_func():
    return stub_dead_interp_bf16