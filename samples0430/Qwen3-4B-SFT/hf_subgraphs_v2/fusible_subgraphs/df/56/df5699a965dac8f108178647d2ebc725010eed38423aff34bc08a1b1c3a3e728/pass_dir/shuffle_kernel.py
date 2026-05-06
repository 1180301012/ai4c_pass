"""
Shared channel-shuffle Triton kernel.

Implements the interleaving pattern:
  cat([x, y], dim=1) -> view(2, C, H, W)
  -> transpose(1, 2) -> contiguous() -> view(2*C, H, W)
  -> chunk(2, dim=1)

  Result[0] = x interleaved in first half  [B, C, H, W]
  Result[1] = y interleaved in second half [B, C, H, W]

grid = (B * C,)  — one program per (batch, source_channel) pair
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 256},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK': 512},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=2),
    ],
    key=['HW'],
)
@triton.jit
def _channel_shuffle_kernel(
    x_ptr, y_ptr, out0_ptr, out1_ptr,
    B, C, HW,
    BLOCK: tl.constexpr,
):
    pid     = tl.program_id(0)
    b       = pid // C
    src_c   = pid % C
    dst_c   = 2 * src_c          # destination channel in the output tensor

    base_src = b * C * HW + src_c * HW
    base_out = b * C * HW + dst_c * HW

    for hw_start in range(0, HW, BLOCK):
        hw_off = hw_start + tl.arange(0, BLOCK)
        mask   = hw_off < HW
        xv = tl.load(x_ptr + base_src + hw_off, mask=mask, other=0.0)
        yv = tl.load(y_ptr + base_src + hw_off, mask=mask, other=0.0)
        tl.store(out0_ptr + base_out + hw_off, xv, mask=mask)
        tl.store(out1_ptr + base_out + hw_off, yv, mask=mask)


@torch.fx.wrap
def shuffler_C20_HW3072(x, y):
    """cat([x,y],1) shuffle → chunk(2) for B*20*64*48 inputs."""
    C  = 20
    HW = 64 * 48   # 3072
    B  = x.shape[0]
    out0 = torch.empty_like(x)
    out1 = torch.empty_like(y)
    _channel_shuffle_kernel[(B * C,)](x, y, out0, out1, B, C, HW)
    return out0, out1


@torch.fx.wrap
def shuffler_C40_HW768(x, y):
    """cat([x,y],1) shuffle → chunk(2) for B*40*32*24 inputs."""
    C  = 40
    HW = 32 * 24   # 768
    B  = x.shape[0]
    out0 = torch.empty_like(x)
    out1 = torch.empty_like(y)
    _channel_shuffle_kernel[(B * C,)](x, y, out0, out1, B, C, HW)
    return out0, out1