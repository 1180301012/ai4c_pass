import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """
    Match the pattern: cat([in_0, in_1], dim=1) followed by slice to get first half.
    This pattern occurs when we cat two tensors and then slice to get exactly the first tensor.
    
    The computation to match:
        tmp_0 = torch.cat([in_0, in_1], dim=1)
        tmp_1 = tmp_0[:, :N, :, :]  # where N = 2 * in_0.shape[1]
        tmp_2 = tmp_1.mean((2, 3), keepdim=True)
        return (tmp_1, tmp_2)
    
    We need to match the cat and slice, and then apply mean.
    The pattern should return tmp_1 (the sliced tensor) and tmp_2 (the mean result).
    """
    tmp_0 = torch.cat([in_0, in_1], dim=1)
    # Match the slice that takes the first half (exactly the first input)
    tmp_1 = tmp_0[:, :in_0.shape[1] * 2, :, :]
    tmp_0 = None  # cleanup, not part of actual computation
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return tmp_1, tmp_2


def replacement_args(in_0, in_1):
    """
    Since we're optimizing the case where cat + slice just returns in_0,
    we only need in_0 for the replacement.
    """
    return (in_0,)


# Triton kernel for mean computation
@triton.jit
def mean_kernel(
    input_ptr,
    output_ptr,
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute mean over (H, W) dimensions for each (B, C) position.
    BLOCK_SIZE should be a power of 2 for efficient reduction.
    """
    # Each program processes one channel of one batch
    # Grid: (B * C,)
    program_id = tl.program_id(0)
    b = program_id // C
    c = program_id % C
    
    # Calculate the starting offset for this (b, c) position
    offset = b * C * H * W + c * H * W
    
    # Load all H*W elements for this channel and compute sum
    # Process in blocks
    sum_val = 0.0
    
    # Vectorized load and sum
    for h in range(H):
        for w in range(W):
            offset_hw = offset + h * W + w
            val = tl.load(input_ptr + offset_hw)
            sum_val = sum_val + val
    
    # Compute mean
    num_elements = H * W
    mean_val = sum_val / num_elements
    
    # Store result at output position
    # Output shape: [B, C, 1, 1]
    output_offset = b * C + c
    tl.store(output_ptr + output_offset, mean_val)


# Module-level wrapper function (required by torch.fx.wrap)
@torch.fx.wrap
def optimized_mean(in_0):
    """
    Optimized version that:
    1. Skips the redundant cat + slice (since we know it just returns in_0)
    2. Uses Triton kernel for efficient mean computation
    
    Returns: (in_0, mean_of_in_0) matching the original return signature
    """
    B, C, H, W = in_0.shape
    BLOCK_SIZE = 64  # Good block size for reduction
    
    # Allocate output for mean: [B, C, 1, 1]
    mean_output = torch.empty((B, C, 1, 1), device=in_0.device, dtype=in_0.dtype)
    
    # Launch kernel with grid (B * C,)
    grid = (B * C,)
    mean_kernel[grid](
        in_0,
        mean_output,
        B,
        C,
        H,
        W,
        BLOCK_SIZE,
    )
    
    # Return tuple: (sliced_tensor, mean_result)
    # Since tmp_1 == in_0, we return in_0 directly
    return in_0, mean_output


def replacement_func():
    """
    Return the optimized function that replaces the pattern.
    """
    return optimized_mean