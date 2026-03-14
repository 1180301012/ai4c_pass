import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Different block sizes for different spatial dimensions
        triton.Config({'BLOCK_SIZE_H': 7, 'BLOCK_SIZE_W': 7}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_H': 8, 'BLOCK_SIZE_W': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_H': 4, 'BLOCK_SIZE_W': 16}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_H': 16, 'BLOCK_SIZE_W': 4}, num_stages=3, num_warps=8),
    ],
    key=['B', 'C', 'H', 'W'],
)
@triton.jit
def fused_relu_add_avgpool_kernel(
    in_0_ptr, in_1_ptr, out_ptr,
    B, C, H, W,
    stride_b, stride_c, stride_h, stride_w,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
):
    """
    Fused kernel that computes:
    1. relu(in_1) + in_0
    2. adaptive_avg_pool2d(result, 1)
    
    Output shape: [B, C, 1, 1]
    """
    # Each program processes one batch and one channel
    batch_idx = tl.program_id(0) // C
    channel_idx = tl.program_id(0) % C
    
    # Check if we're within bounds
    if batch_idx >= B:
        return
    
    # Compute the sum over all spatial positions for this batch and channel
    sum_val = 0.0
    
    # Iterate over spatial dimensions
    for h_idx in range(H):
        for w_idx in range(W):
            # Compute offsets
            offset = batch_idx * stride_b + channel_idx * stride_c + h_idx * stride_h + w_idx * stride_w
            
            # Load in_0 and in_1
            val_0 = tl.load(in_0_ptr + offset)
            val_1 = tl.load(in_1_ptr + offset)
            
            # Apply ReLU to in_1
            relu_val = tl.maximum(val_1, 0.0)
            
            # Add in_0 + relu(in_1)
            fused_val = val_0 + relu_val
            
            # Accumulate for average
            sum_val += fused_val
    
    # Compute average (divide by total spatial elements)
    num_elements = H * W
    avg_val = sum_val / num_elements
    
    # Store result
    out_offset = batch_idx * C + channel_idx
    tl.store(out_ptr + out_offset, avg_val)


@torch.fx.wrap
def fused_relu_add_avgpool_wrapper(in_0: torch.Tensor, in_1: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function that launches the fused Triton kernel.
    
    in_0: [B, C, H, W]
    in_1: [B, C, H, W]
    output: [B, C, 1, 1]
    """
    B, C, H, W = in_0.shape
    
    # Create output tensor
    out = torch.empty((B, C, 1, 1), device=in_0.device, dtype=in_0.dtype)
    
    # Launch kernel - one program per (batch, channel) pair
    grid = (B * C,)
    
    fused_relu_add_avgpool_kernel[grid](
        in_0, in_1, out,
        B, C, H, W,
        in_0.stride(0), in_0.stride(1), in_0.stride(2), in_0.stride(3),
    )
    
    return out


def pattern(in_0, in_1):
    """
    Match the computation pattern:
    1. relu(in_1)
    2. add result with in_0
    3. adaptive_avg_pool2d with output_size=1
    """
    tmp_0 = torch.nn.functional.relu(in_1, inplace=False)
    tmp_1 = tmp_0 + in_0
    tmp_2 = torch.nn.functional.adaptive_avg_pool2d(tmp_1, 1)
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fused_relu_add_avgpool_wrapper