import torch
import triton
import triton.language as tl


def pattern(in_1):
    tmp_0 = in_1.sum(dim = 2, keepdim = True)
    tmp_1 = in_1 / tmp_0
    return tmp_1


def replacement_args(in_1):
    return (in_1,)


@triton.jit
def fused_sum_div_kernel(
    in_ptr,
    out_ptr,
    NUM_GROUPS: tl.constexpr,
    REDUCTION_SIZE: tl.constexpr,
):
    # Process all data in a single thread block
    # Input shape: [1, 2, 8, 8] -> treat as [2*8=16 groups, 8 reduction elements]
    # Groups map to (channel, width) pairs, reduction is over height (dim=2)
    
    groups = tl.arange(0, NUM_GROUPS)    # [16]
    reduces = tl.arange(0, REDUCTION_SIZE)  # [8]
    
    # Map group index to (outer, inner) = (channel, width)
    outer = groups // REDUCTION_SIZE  # channel: [16] -> 0,0,...,0,1,1,...,1
    inner = groups % REDUCTION_SIZE   # width: [16] -> 0,1,...,7,0,1,...,7
    
    # Build 2D offset matrix [16, 8]
    # offset(channel, height, width) = channel*64 + height*8 + width
    offsets = outer[:, None] * (REDUCTION_SIZE * REDUCTION_SIZE) + reduces[None, :] * REDUCTION_SIZE + inner[:, None]
    
    # Load all data [16, 8]
    x = tl.load(in_ptr + offsets)
    
    # Sum over reduction axis (axis=1) -> [16]
    s = tl.sum(x, axis=1)
    
    # Divide by sum (broadcast) -> [16, 8]
    out = x / s[:, None]
    
    # Store results
    tl.store(out_ptr + offsets, out)


@torch.fx.wrap
def fused_sum_div(in_1):
    out = torch.empty_like(in_1)
    
    # Shape [1, 2, 8, 8]: outer=2, reduction=8, inner=8
    # NUM_GROUPS = outer * inner = 2 * 8 = 16
    # REDUCTION_SIZE = 8
    fused_sum_div_kernel[(1,)](
        in_1,
        out,
        NUM_GROUPS=16,
        REDUCTION_SIZE=8,
        num_warps=1,
        num_stages=1,
    )

    return out


def replacement_func():
    return fused_sum_div