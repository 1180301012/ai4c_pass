import torch
import triton
import triton.language as tl


# Match the broadcast scale-multiply: in_3[B,C,H,W] * tmp_4[B,C,1,1]
# This pattern is generic and matches all six target graphs regardless of
# batch size or spatial resolution.
def pattern(in_3, tmp_4):
    tmp_5 = in_3 * tmp_4
    return tmp_5


def replacement_args(in_3, tmp_4):
    return (in_3, tmp_4)


@triton.autotune(
    configs=[
        # Pipelined loop configs: each CTA iterates over all HW tiles with
        # software prefetching (num_stages) to better hide memory latency.
        triton.Config({'BLOCK_SIZE': 256,  'NUM_STAGES': 1}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512,  'NUM_STAGES': 1}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512,  'NUM_STAGES': 2}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024, 'NUM_STAGES': 1}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024, 'NUM_STAGES': 2}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024, 'NUM_STAGES': 3}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048, 'NUM_STAGES': 1}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048, 'NUM_STAGES': 2}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096, 'NUM_STAGES': 1}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096, 'NUM_STAGES': 2}, num_warps=16),
    ],
    key=['HW'],
)
@triton.jit
def _broadcast_scale_mul_kernel(
    in3_ptr,    # [B, C, H, W] contiguous
    scale_ptr,  # [B*C] flattened from [B, C, 1, 1]
    out_ptr,    # [B, C, H, W] contiguous
    HW,         # H * W
    BLOCK_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    # 1D grid: one program per (b, c) pair.
    # Each program loops over all HW spatial elements in BLOCK_SIZE tiles.
    # The tl.range num_stages argument enables software pipelining:
    # tile i+1 is prefetched while tile i is being processed.
    bc_pid = tl.program_id(0)

    # Load the scale scalar for this (b, c) pair – used by all tiles.
    scale = tl.load(scale_ptr + bc_pid)
    base  = bc_pid * HW

    for start in tl.range(0, HW, BLOCK_SIZE, num_stages=NUM_STAGES):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < HW
        x   = tl.load(in3_ptr + base + offs, mask=mask)
        tl.store(out_ptr + base + offs, x * scale, mask=mask)


@torch.fx.wrap
def fast_broadcast_scale_mul(in_3, tmp_4):
    B, C, H, W = in_3.shape
    HW  = H * W
    BC  = B * C

    # For small tensors the Triton/wrapper overhead exceeds the compute
    # savings.  Delegate to PyTorch's native broadcast multiply instead.
    if BC * HW < 10_000_000:
        return in_3 * tmp_4

    out   = torch.empty_like(in_3)
    # Flatten tmp_4 [B, C, 1, 1] -> [B*C] for O(1) indexing per program
    scale = tmp_4.reshape(BC)

    # 1D grid: one program per (b,c) pair; each loops over spatial tiles
    _broadcast_scale_mul_kernel[(BC,)](
        in_3, scale, out,
        HW,
    )
    return out


def replacement_func():
    return fast_broadcast_scale_mul