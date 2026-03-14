import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """
    Pattern: add -> silu -> adaptive_avg_pool2d(1) -> flatten(1,-1) -> dropout(p=0)
    This fuses the entire computation chain into a single optimized kernel.
    """
    tmp_0 = in_1 + in_0
    tmp_1 = torch.nn.functional.silu(tmp_0, inplace=False)
    tmp_2 = torch.nn.functional.adaptive_avg_pool2d(tmp_1, 1)
    tmp_3 = tmp_2.flatten(1, -1)
    tmp_4 = torch.nn.functional.dropout(tmp_3, 0.0, False, False)
    return tmp_4


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_add_silu_pool_kernel(
    in_0_ptr, in_1_ptr, out_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_C: tl.constexpr,
):
    """
    Fused kernel that computes:
    1. out = silu(in_0 + in_1)  [element-wise]
    2. out = mean(out, axis=[2,3])  [spatial average]
    
    We process in blocks: each program processes (BLOCK_SIZE_B, BLOCK_SIZE_C) channels
    across all spatial positions.
    """
    # Get batch and channel indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Calculate pointers for this batch and channel
    in_0_offset = batch_idx * C * H * W + channel_idx * H * W
    in_1_offset = batch_idx * C * H * W + channel_idx * H * W
    out_offset = batch_idx * C + channel_idx
    
    # Load in_0 and in_1, compute sum, then apply silu
    # Then accumulate for average
    sum_val = 0.0
    
    # Iterate over spatial dimensions
    for h in range(H):
        for w in range(W):
            # Load in_0[in_0_offset + h*W + w]
            in_0_ptr_idx = in_0_ptr + in_0_offset + h * W + w
            in_1_ptr_idx = in_1_ptr + in_1_offset + h * W + w
            
            x0 = tl.load(in_0_ptr_idx)
            x1 = tl.load(in_1_ptr_idx)
            
            # Element-wise addition
            x = x0 + x1
            
            # SiLU: x * sigmoid(x) = x * (1 / (1 + exp(-x)))
            sigmoid = 1.0 / (1.0 + tl.exp(-x))
            silu_val = x * sigmoid
            
            # Accumulate for spatial average
            sum_val += silu_val
    
    # Compute mean across spatial dimensions
    num_spatial = H * W
    mean_val = sum_val / num_spatial
    
    # Store result
    tl.store(out_ptr + out_offset, mean_val)


@torch.fx.wrap
def fused_add_silu_pool_wrapper(in_0, in_1):
    """
    Wrapper function that launches the fused kernel.
    Input shapes: [B, C, H, W]
    Output shape: [B, C]
    """
    B, C, H, W = in_0.shape
    
    # Allocate output
    out = torch.empty((B, C), dtype=torch.float32, device=in_0.device)
    
    # Define block sizes - process multiple channels per program for efficiency
    # Each program processes all spatial positions for one channel
    BLOCK_SIZE_B = 1
    BLOCK_SIZE_C = 1
    
    # Grid: (B, C) - each (batch, channel) pair processed by one program
    grid = (B, C)
    
    fused_add_silu_pool_kernel[grid](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        B=B, C=C, H=H, W=W,
        BLOCK_SIZE_B=BLOCK_SIZE_B,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
    )
    
    return out


def replacement_func():
    return fused_add_silu_pool_wrapper