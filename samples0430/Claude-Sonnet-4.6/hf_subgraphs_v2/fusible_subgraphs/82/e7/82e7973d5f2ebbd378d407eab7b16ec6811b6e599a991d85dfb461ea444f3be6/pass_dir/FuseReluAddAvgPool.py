import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: relu(in_1) + in_0  →  adaptive_avg_pool2d(result, 1)
# This matches the exact computation in all target model graphs.
# ---------------------------------------------------------------------------

def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.relu(in_1, inplace=False)
    tmp_1 = tmp_0 + in_0
    tmp_2 = torch.nn.functional.adaptive_avg_pool2d(tmp_1, 1)
    return (tmp_2,)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ---------------------------------------------------------------------------
# Triton kernel
# Each program handles exactly one (B, C) slice of size H*W.
# We fuse relu + add into the load path and reduce-sum over the slice,
# then divide by H*W to produce the global average.
# Float32 accumulation is used for numerical stability regardless of input
# dtype; Triton's tl.store auto-casts to the output pointer's element type.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64},   num_warps=2),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def fused_relu_add_avgpool_kernel(
    in0_ptr,   # [B, C, H, W] contiguous
    in1_ptr,   # [B, C, H, W] contiguous
    out_ptr,   # [B, C, 1, 1] contiguous
    HW,        # H * W  (runtime scalar)
    BLOCK_SIZE: tl.constexpr,   # must be >= HW, a power of 2
):
    # One program per (b, c) pair
    pid    = tl.program_id(0)
    base   = pid * HW
    offs   = tl.arange(0, BLOCK_SIZE)
    mask   = offs < HW

    # Load inputs; masked positions contribute 0.0 to the sum
    in0 = tl.load(in0_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    in1 = tl.load(in1_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

    # Fused relu + add
    relu_val = tl.maximum(in1, 0.0)
    fused    = relu_val + in0

    # Global average
    total = tl.sum(fused, axis=0)
    avg   = total / HW

    # Store scalar result (Triton auto-casts to out_ptr element type)
    tl.store(out_ptr + pid, avg)


# ---------------------------------------------------------------------------
# Wrapper (must be @torch.fx.wrap so the FX rewriter can call it)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_relu_add_avgpool(in_0, in_1):
    B, C, H, W = in_0.shape
    HW = H * W
    BC = B * C

    out = torch.empty((B, C, 1, 1), dtype=in_0.dtype, device=in_0.device)

    # BLOCK_SIZE must be >= HW; use the next power-of-2 so the autotune key
    # 'HW' selects the right compiled variant.
    BLOCK_SIZE = triton.next_power_of_2(HW)

    fused_relu_add_avgpool_kernel[(BC,)](
        in_0,
        in_1,
        out,
        HW,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return (out,)


def replacement_func():
    return fused_relu_add_avgpool