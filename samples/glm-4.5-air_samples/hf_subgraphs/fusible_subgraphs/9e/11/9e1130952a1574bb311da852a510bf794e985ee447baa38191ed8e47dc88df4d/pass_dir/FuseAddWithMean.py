import torch
import triton
import triton.language as tl

@torch.fx.wrap
def fused_add_mean(x1, x2):
    # Get input dimensions
    N, C, H, W = x1.shape
    assert x2.shape == x1.shape, "Inputs must have same shape"
    
    # Cast to proper dtype if needed
    x1 = x1.to(torch.float32)
    x2 = x2.to(torch.float32)
    
    # Create output tensor
    output = torch.empty((C,), dtype=torch.float32, device=x1.device)
    
    # Launch kernel
    grid = (C,)
    fused_add_mean_kernel[grid](
        x1,
        x2,
        output,
        N, C, H, W,
        BLOCK_SIZE_HW=256,
    )
    
    return output

def pattern(x1, x2):
    # Element-wise addition followed by mean reduction
    sum_x = x1 + x2
    mean_val = sum_x.mean((2, 3), keepdim=False)
    return mean_val

def replacement_args(x1, x2):
    return (x1, x2)

@triton.jit
def fused_add_mean_kernel(
    x1_ptr,
    x2_ptr,
    output_ptr,
    N, C, H, W,
    BLOCK_SIZE_HW: tl.constexpr,
):
    # Each program handles a specific channel
    c = tl.program_id(0)
    
    # Compute total elements for mean normalization
    total_hw = H * W
    
    # Calculate how many blocks needed for this channel
    num_blocks = (total_hw + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW
    
    # Accumulators for mean computation
    sum_val = 0.0
    
    # Process spatial elements across all blocks for this channel
    for block_idx in range(num_blocks):
        # Calculate start offset for this block
        block_start = block_idx * BLOCK_SIZE_HW
        
        # Generate constant offsets within this block
        offsets = tl.arange(0, BLOCK_SIZE_HW)
        
        # Convert to absolute offsets and mask out of bounds
        abs_offsets = block_start + offsets
        mask = abs_offsets < total_hw
        
        # Load data for this channel across all spatial positions
        x1_ptr_c_offset = x1_ptr + c * (H * W) + abs_offsets
        x2_ptr_c_offset = x2_ptr + c * (H * W) + abs_offsets
        
        x1 = tl.load(x1_ptr_c_offset, mask=mask, other=0.0)
        x2 = tl.load(x2_ptr_c_offset, mask=mask, other=0.0)
        
        # Add and accumulate
        sum_x = x1 + x2
        sum_val += tl.sum(sum_x)
    
    # Compute mean and store
    mean_val = sum_val / total_hw
    tl.store(output_ptr + c, mean_val)

def replacement_func():
    return fused_add_mean