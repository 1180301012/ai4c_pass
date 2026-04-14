import torch
import triton
import triton.language as tl

# Pattern that matches the specific double unsqueeze + broadcasting computation
# This captures: scale.unsqueeze(-1).unsqueeze(-1) * residual + original
def pattern(scale_tensor, residual_tensor, original_tensor):
    # Double unsqueeze for broadcasting: [48] -> [48, 1, 1]
    expanded_scale = scale_tensor.unsqueeze(-1).unsqueeze(-1)
    
    # Broadcasting multiplication with residual
    scaled_residual = expanded_scale * residual_tensor
    
    # Add back to original (similar to residual connection)
    result = original_tensor + scaled_residual
    
    return result

def replacement_args(scale_tensor, residual_tensor, original_tensor):
    return (scale_tensor, residual_tensor, original_tensor)

# Optimized Triton kernel for broadcasting operation
@triton.jit
def broadcasting_scale_kernel(
    scale_ptr,
    residual_ptr,
    original_ptr,
    out_ptr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    batch_size: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Program IDs for efficient 3D grid processing
    batch_pid = tl.program_id(0)
    spatial_pid = tl.program_id(1)
    channel_pid = tl.program_id(2)
    
    # Bounds checking
    if batch_pid >= batch_size or channel_pid >= channels:
        return
    
    spatial_size = height * width
    spatial_start = spatial_pid * BLOCK_SIZE_N
    
    # Create offset range for this work item
    offsets = spatial_start + tl.arange(0, BLOCK_SIZE_N)
    mask = offsets < spatial_size
    
    # Load scale value for this channel
    scale_val = tl.load(scale_ptr + channel_pid)
    
    # Calculate indices for loading data
    spatial_indices = offsets
    channel_indices = tl.full([BLOCK_SIZE_N], channel_pid, dtype=tl.int32)
    batch_indices = tl.full([BLOCK_SIZE_N], batch_pid, dtype=tl.int32)
    
    # Calculate linear indices: (C * HW + spatial_idx) * batch_size + batch_idx
    linear_indices = (channel_indices * spatial_size + spatial_indices) * batch_size + batch_indices
    
    # Load residual and original values with bounds checking
    residual_vals = tl.load(residual_ptr + linear_indices, mask=mask, other=0.0)
    original_vals = tl.load(original_ptr + linear_indices, mask=mask, other=0.0)
    
    # Apply broadcasting operation: scale * residual + original
    # Since scale broadcasted from [C] to [C,1,1] affects all spatial positions
    results = original_vals + scale_val * residual_vals
    
    # Store results
    tl.store(out_ptr + linear_indices, results, mask=mask)

# Optimized kernel with better tuning parameters
@triton.jit
def broadcasting_scale_kernel_optimized(
    scale_ptr,
    residual_ptr,
    original_ptr,
    out_ptr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    batch_size: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Program IDs for efficient 3D grid processing
    batch_pid = tl.program_id(0)
    spatial_pid = tl.program_id(1)
    channel_pid = tl.program_id(2)
    
    # Bounds checking
    if batch_pid >= batch_size or channel_pid >= channels:
        return
    
    spatial_size = height * width
    spatial_start = spatial_pid * BLOCK_SIZE_N
    
    # Create offset range for this work item
    offsets = spatial_start + tl.arange(0, BLOCK_SIZE_N)
    mask = offsets < spatial_size
    
    # Load scale value for this channel
    scale_val = tl.load(scale_ptr + channel_pid)
    
    # Calculate linear indices directly for better memory access
    idx = (channel_pid * spatial_size + offsets) * batch_size + batch_pid
    
    # Load residual and original values with bounds checking
    residual_vals = tl.load(residual_ptr + idx, mask=mask, other=0.0)
    original_vals = tl.load(original_ptr + idx, mask=mask, other=0.0)
    
    # Apply broadcasting operation: scale * residual + original
    results = original_vals + scale_val * residual_vals
    
    # Store results
    tl.store(out_ptr + idx, results, mask=mask)

# Kernel wrapper with better optimization
@torch.fx.wrap 
def optimized_broadcasting_add(scale, residual, original):
    # Check if we have 4D tensors that benefit from GPU optimization
    if (len(scale.shape) == 1 and len(residual.shape) == 4 and 
        len(original.shape) == 4 and residual.shape == original.shape and
        residual.numel() > 4096):  # Only optimize for larger tensors
            
        batch_size, channels, height, width = residual.shape
        
        # Prepare output tensor
        out = torch.empty_like(residual)
        
        # Optimized block size for better GPU utilization
        BLOCK_SIZE_N = 512  # Balance between parallelism and overhead
        spatial_size = height * width
        num_spatial_blocks = (spatial_size + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
        grid = (batch_size, min(num_spatial_blocks, 1024), channels)  # Limit spatial blocks
        
        # Launch optimized kernel
        broadcasting_scale_kernel_optimized[grid](
            scale, residual, original, out,
            channels, height, width, batch_size,
            BLOCK_SIZE_N
        )
        
        return out
    else:
        # Fallback to original operation for unsupported tensor shapes or small tensors
        expanded_scale = scale.unsqueeze(-1).unsqueeze(-1)
        scaled_residual = expanded_scale * residual
        return original + scaled_residual

def replacement_func():
    return optimized_broadcasting_add