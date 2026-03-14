import torch
import triton
import triton.language as tl

# Pattern matching function - Case 1: with keepdim, returns (silu, mean)
def pattern(in_0):
    """Match SILU + Mean reduction pattern with keepdim=True"""
    # SILU: x * sigmoid(x) - direct implementation
    tmp_sigmoid = torch.sigmoid(in_0)
    tmp_0 = in_0 * tmp_sigmoid
    tmp_1 = tmp_0.mean((2, 3), keepdim=True)
    return (tmp_0, tmp_1)

# Pattern matching function - Case 2: without keepdim, returns (mean, silu)  
def pattern2(in_0):
    """Match SILU + Mean reduction pattern with keepdim=False"""
    # SILU: x * sigmoid(x) - direct implementation  
    tmp_sigmoid = torch.sigmoid(in_0)
    tmp_0 = in_0 * tmp_sigmoid
    tmp_1 = tmp_0.mean((2, 3))
    return (tmp_1, tmp_0)

# Argument extraction function  
def replacement_args(in_0):
    """Extract input tensor"""
    return (in_0,)

# Custom Triton kernel that fuses SILU and mean reduction
@triton.jit
def silu_mean_kernel(
    x_ptr,
    silu_out_ptr,
    mean_out_ptr,
    batch_size,
    channels,
    height,
    width,
    keepdim: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused SILU + Mean reduction kernel"""
    pid_batch = tl.program_id(0)
    pid_channel = tl.program_id(1)
    
    # Calculate input tensor position
    x_offset = pid_batch * (channels * height * width) + pid_channel * (height * width)
    
    # Load input block
    x_base_ptr = x_ptr + x_offset
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < height * width
    
    x = tl.load(x_base_ptr + offsets, mask=mask, other=0.0)
    
    # Compute SILU: x * sigmoid(x)
    x_sigmoid = 1.0 / (1.0 + tl.exp(-x))
    silu_out = x * x_sigmoid
    
    # Store SILU output
    silu_base_ptr = silu_out_ptr + x_offset
    tl.store(silu_base_ptr + offsets, silu_out, mask=mask)
    
    # Compute mean reduction over spatial dimensions
    # First, do local sum for each thread
    local_sum = tl.sum(silu_out, axis=0)
    
    # Barrier for all threads in this channel
    tl.device_barrier()
    
    # If we have multiple blocks per channel, do global reduction
    if BLOCK_CHANNELS > 1:
        # Store local sums to shared memory
        shared_ptr = tl.arange(0, BLOCK_CHANNELS) * (height * width // BLOCK_SIZE)
        tl.store(shared_ptr + tl.program_id(2), local_sum, eviction_policy='evict_last')
        
        # Barrier for all threads in this channel
        tl.device_barrier()
        
        # Thread 0 does final reduction
        if tl.program_id(2) == 0:
            final_sum = 0.0
            for i in range(BLOCK_CHANNELS):
                shared_i_ptr = i * (height * width // BLOCK_SIZE)
                val = tl.load(shared_ptr + shared_i_ptr)
                final_sum += val
            
            # Compute final mean
            total_elements = height * width
            mean_val = final_sum / total_elements
            
            # Store mean result
            if keepdim:
                mean_out_base_ptr = mean_out_ptr + (pid_batch * channels + pid_channel) * 4
                # Store mean value in [1,1] spatial layout
                tl.store(mean_out_base_ptr + 0, mean_val)
                tl.store(mean_out_base_ptr + 1, mean_val)  
                tl.store(mean_out_base_ptr + 2, mean_val)
                tl.store(mean_out_base_ptr + 3, mean_val)
            else:
                mean_out_base_ptr = mean_out_ptr + (pid_batch * channels + pid_channel)
                tl.store(mean_out_base_ptr, mean_val)
    else:
        # Single block per channel, direct computation
        total_elements = height * width  
        mean_val = local_sum / total_elements
        
        # Store mean result
        if keepdim:
            mean_out_base_ptr = mean_out_ptr + (pid_batch * channels + pid_channel) * 4
            # Store mean value in [1,1] spatial layout
            tl.store(mean_out_base_ptr + 0, mean_val)
            tl.store(mean_out_base_ptr + 1, mean_val)
            tl.store(mean_out_base_ptr + 2, mean_val) 
            tl.store(mean_out_base_ptr + 3, mean_val)
        else:
            mean_out_base_ptr = mean_out_ptr + (pid_batch * channels + pid_channel)
            tl.store(mean_out_base_ptr, mean_val)

# Wrapper function for keepdim=True case (returns silu, mean)
@torch.fx.wrap
def fused_silu_mean_keepdim(x):
    """Fused SILU + Mean reduction with keepdim=True"""
    return _fused_silu_mean_kernel(x, keepdim=True)

# Wrapper function for keepdim=False case (returns mean, silu) 
@torch.fx.wrap  
def fused_silu_mean_no_keepdim(x):
    """Fused SILU + Mean reduction with keepdim=False"""
    silu_out, mean_out = _fused_silu_mean_kernel(x, keepdim=False)
    return mean_out, silu_out

# Core fused SILU + Mean reduction implementation
def _fused_silu_mean_kernel(x, keepdim=True):
    """Fused SILU + Mean reduction operation"""
    if x.dim() != 4:
        raise ValueError("Input must be 4D tensor [B, C, H, W]")
    
    batch_size, channels, height, width = x.shape
    
    # Allocate output tensors
    silu_out = torch.empty_like(x)
    
    if keepdim:
        mean_out = torch.empty(batch_size, channels, 1, 1, device=x.device, dtype=x.dtype)
    else:
        mean_out = torch.empty(batch_size, channels, device=x.device, dtype=x.dtype)
    
    # Choose block sizes based on tensor dimensions
    BLOCK_SIZE = min(1024, height * width)
    BLOCK_CHANNELS = 1  # Use one block per channel for simplicity and better parallelism
    
    # Calculate grid dimensions
    num_batches = batch_size
    num_channels = channels
    num_blocks_per_channel = (height * width + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    if num_blocks_per_channel > 1:
        grid = (num_batches, num_channels, num_blocks_per_channel)
    else:
        grid = (num_batches, num_channels)
    
    # Launch kernel
    silu_mean_kernel[grid](
        x,
        silu_out,
        mean_out,
        batch_size,
        channels, 
        height,
        width,
        keepdim,
        BLOCK_CHANNELS,
        BLOCK_SIZE
    )
    
    return silu_out, mean_out

# Replacement function for pattern matching
def replacement_func():
    """Returns the appropriate fused function based on which pattern matches"""
    # For now, default to the keepdim=True case
    return fused_silu_mean_keepdim