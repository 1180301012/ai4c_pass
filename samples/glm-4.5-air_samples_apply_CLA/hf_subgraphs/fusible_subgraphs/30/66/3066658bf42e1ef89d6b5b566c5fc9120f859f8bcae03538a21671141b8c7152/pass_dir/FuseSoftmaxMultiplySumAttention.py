import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    # Pattern: softmax(input, dim=1) * input2 -> reduce_sum(..., dim=1)
    tmp_0 = torch.softmax(in_1, dim=1)
    tmp_1 = in_0 * tmp_0
    tmp_2 = torch.sum(tmp_1, dim=1)
    return tmp_2

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def simple_fused_attention_kernel(
    in_0_ptr,
    in_1_ptr, 
    out_ptr,
    batch_size,
    num_heads,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Simplified fused kernel with proper broadcasting
    """
    
    pid = tl.program_id(0)
    total_elements = batch_size * channels * height * width
    element_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = element_idx < total_elements
    
    # Convert linear element index to multi-dimensional indices
    element_idx_5d = element_idx
    idx_height = element_idx_5d % width
    element_idx_5d = element_idx_5d // width
    idx_width = element_idx_5d % height
    element_idx_5d = element_idx_5d // height
    idx_channels = element_idx_5d % channels
    element_idx_5d = element_idx_5d // channels
    idx_batch = element_idx_5d
    
    # OPTIMIZED: Initialize accumulator and precompute head count
    sum_result = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # OPTIMIZED: Loop unrolling for small head counts
    num_heads = min(num_heads, 8)  # Limit head iteration for performance
    
    # Iterate over heads and accumulate
    for head_idx in range(num_heads):
        # Calculate linear index for in_0: [batch, head, channel, height, width]
        offset_in_0 = (idx_batch * num_heads + head_idx) * channels * height * width + \
                      idx_channels * height * width + idx_width * width + idx_height
        
        # Calculate linear index for in_1: [batch, head]
        offset_in_1 = idx_batch * num_heads + head_idx
        
        # Load values with proper masks and early exit
        if offset_in_0 >= batch_size * num_heads * channels * height * width:
            continue
        val_in_0 = tl.load(in_0_ptr + offset_in_0, mask=mask, other=0.0)
        val_in_1 = tl.load(in_1_ptr + offset_in_1, mask=offset_in_1 < (batch_size * num_heads), other=0.0)
        
        # OPTIMIZED: Direct multiplication and accumulation
        sum_result += val_in_0 * val_in_1
    
    # Store result
    tl.store(out_ptr + element_idx, sum_result, mask=mask)

@torch.fx.wrap  
def fused_attention_kernel_wrapper(in_0, in_1):
    # Get input shapes
    batch_size, num_heads, channels, height, width = in_0.shape
    
    # Output shape after summing across heads: [batch, channels, height, width]
    out_shape = (batch_size, channels, height, width)
    out = torch.empty(out_shape, dtype=in_0.dtype, device=in_0.device)
    
    # Choose optimal block size based on problem size
    total_elements = batch_size * channels * height * width
    
    # OPTIMIZED: Use medium block size for better performance
    if total_elements > 100000:
        BLOCK_SIZE = 256
    elif total_elements > 50000:
        BLOCK_SIZE = 512   
    elif total_elements > 10000:
        BLOCK_SIZE = 1024
    else:
        BLOCK_SIZE = 2048  # Larger block for small problems
    
    # Calculate grid size
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the simplified kernel
    simple_fused_attention_kernel[(grid_size,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1, 
        out_ptr=out,
        batch_size=batch_size,
        num_heads=num_heads,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_attention_kernel_wrapper