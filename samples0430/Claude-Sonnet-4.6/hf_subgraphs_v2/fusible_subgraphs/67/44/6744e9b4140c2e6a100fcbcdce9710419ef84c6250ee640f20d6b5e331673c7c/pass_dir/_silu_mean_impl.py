"""Shared Triton kernel for fused SiLU + spatial mean reduction."""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}),
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
    ],
    key=['HW'],
)
@triton.jit
def _silu_mean_kernel(
    x_ptr,          # [B, C, H, W] input (read)
    out_silu_ptr,   # [B, C, H, W] output (SiLU values)
    out_mean_ptr,   # [B, C]       output (channel means)
    HW,             # H * W (runtime)
    BLOCK_SIZE: tl.constexpr,
):
    """One program per (b, c) pair. Compute SiLU and accumulate mean."""
    pid   = tl.program_id(0)
    base  = pid * HW
    acc   = 0.0

    for start in range(0, HW, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask    = offsets < HW
        x       = tl.load(x_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
        silu_x  = x * tl.sigmoid(x)
        tl.store(out_silu_ptr + base + offsets, silu_x, mask=mask)
        acc = acc + tl.sum(silu_x, axis=0)

    tl.store(out_mean_ptr + pid, acc / HW)


@torch.fx.wrap
def fused_silu_mean(in_1):
    """Apply SiLU in-place and compute spatial mean via Triton."""
    B, C, H, W = in_1.shape
    HW = H * W
    out_silu = torch.empty_like(in_1)
    # Return mean as [B, C]; the remaining view in the graph reshapes to [1,1,C]
    out_mean = torch.empty((B, C), dtype=in_1.dtype, device=in_1.device)
    _silu_mean_kernel[(B * C,)](in_1, out_silu, out_mean, HW=HW)
    return (out_silu, out_mean)