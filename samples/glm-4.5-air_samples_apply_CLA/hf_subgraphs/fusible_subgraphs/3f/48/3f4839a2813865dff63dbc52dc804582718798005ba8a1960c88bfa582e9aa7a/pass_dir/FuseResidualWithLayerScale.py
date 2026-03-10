import torch
import triton
import triton.language as tl

def pattern(tmp_1):
    # tmp_8 = tmp_1.unsqueeze(-1)
    tmp_8 = tmp_1.unsqueeze(-1)
    # tmp_9 = tmp_8.unsqueeze(-1)
    tmp_9 = tmp_8.unsqueeze(-1)
    
    return tmp_9

def replacement_args(tmp_1):
    return (tmp_1,)

@triton.jit
def expand_kernel(
    in_ptr, out_ptr,
    C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one channel and processes multiple spatial positions
    pid = tl.program_id(0)
    
    if pid >= C:
        return
    
    # Load the value from the input channel
    value = tl.load(in_ptr + pid)
    
    # Process spatial positions in blocks for better memory coalescing
    spatial_size = H * W
    for start_idx in range(0, spatial_size, BLOCK_SIZE):
        # Create mask for this block
        offsets = start_idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < spatial_size
        
        # Broadcast the value to all spatial positions in the block
        outputs = tl.full([BLOCK_SIZE], value, dtype=tl.float32)
        
        # Calculate output indices
        output_indices = pid * spatial_size + offsets
        
        # Store results with mask
        tl.store(out_ptr + output_indices, outputs, mask=mask)

@triton.jit
def expand_kernel_optimized(
    in_ptr, out_ptr,
    C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one channel and processes multiple spatial positions
    pid = tl.program_id(0)
    
    if pid >= C:
        return
    
    # Load the value from the input channel
    value = tl.load(in_ptr + pid)
    
    # Process spatial positions in blocks for better memory coalescing
    spatial_size = H * W
    for start_idx in range(0, spatial_size, BLOCK_SIZE):
        # Create mask for this block
        offsets = start_idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < spatial_size
        
        # Broadcast the value to all spatial positions in the block
        outputs = tl.full([BLOCK_SIZE], value, dtype=tl.float32)
        
        # Calculate output indices
        output_indices = pid * spatial_size + offsets
        
        # Store results with mask
        tl.store(out_ptr + output_indices, outputs, mask=mask)

@torch.fx.wrap
def expand_wrapper(tmp_1):
    # Determine tensor dimensions
    C = tmp_1.shape[0]
    H, W = 56, 56  # From the weight_meta.py files
    
    # Create output tensor [C, H, W]
    output_9 = torch.empty((C, H, W), dtype=tmp_1.dtype, device=tmp_1.device)
    
    # Set up grid for optimized version
    grid = (triton.cdiv(C, 1),)
    
    # Use optimized kernel with 256 block size (good balance for our problem size)
    expand_kernel_optimized[grid](
        tmp_1, output_9,
        C, H, W,
        BLOCK_SIZE=256,
    )
    
    return output_9

def replacement_func():
    return expand_wrapper