import torch
import triton
import triton.language as tl

# Pattern matching function for the 1, 128 case
def pattern(tmp_7):
    """Pattern matches adaptive_avg_pool2d + view fusion for 1,128 output"""
    tmp_8 = torch.nn.functional.adaptive_avg_pool2d(tmp_7, 1)
    tmp_9 = tmp_8.view(1, 128)
    return tmp_9

# Argument extraction function  
def replacement_args(tmp_7):
    return (tmp_7, (1, 128))

# Optimized Triton kernel for fused global average pooling + reshape
@triton.jit
def fused_global_pool_view_kernel(
    # Input tensor
    input_ptr,
    # Output tensor  
    output_ptr,
    # Input dimensions
    batch_size,
    channels,
    height,
    width,
    # View shape parameters  
    view_dim0,
    view_dim1,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one element in the flattened output
    output_idx = tl.program_id(0)
    
    # Calculate which batch and channel this output corresponds to
    batch_idx = output_idx // (view_dim0 * view_dim1) 
    ch_idx = (output_idx % (view_dim0 * view_dim1)) // view_dim1
    
    # Check bounds
    batch_mask = batch_idx < batch_size
    ch_mask = ch_idx < channels
    
    if not (batch_mask & ch_mask):
        return
    
    # Compute global average for this batch and channel
    # For adaptive_avg_pool2d with size 1, we average over all spatial elements
    sum_val = 0.0
    element_count = height * width
    
    # Create offsets for all spatial elements of this batch and channel
    # Layout: [batch, channel, height, width] -> flatten to 1D
    base_offset = batch_idx * channels * height * width + ch_idx * height * width
    
    # Process spatial elements in blocks
    for offset_start in range(0, element_count, BLOCK_SIZE):
        # Use fixed-size arange with masking for boundaries
        local_offsets = tl.arange(0, BLOCK_SIZE)
        global_offsets = base_offset + offset_start + local_offsets
        
        # Create masks for boundaries
        valid_mask = (local_offsets < (element_count - offset_start))
        tensor_mask = global_offsets < batch_size * channels * height * width
        
        # Load spatial elements with proper masking
        x = tl.load(input_ptr + global_offsets, mask=valid_mask & tensor_mask, other=0.0)
        sum_val += tl.sum(x)
    
    # Compute global mean
    global_mean = sum_val / element_count
    
    # Store in flattened output (output_idx is already correct for flattened layout)
    # For view(128, 128) or view(1, 128), we need to distribute the result
    if view_dim0 == 1:
        # Case: view(1, 128) - output_idx should be batch_idx * 128 + ch_idx
        final_idx = batch_idx * channels + ch_idx
    else:
        # Case: view(128, 128) - this suggests batch_size=1, channels=128
        final_idx = ch_idx
    
    # Store the result directly at the output_idx (flattened view layout)
    tl.store(output_ptr + output_idx, global_mean)

# Kernel wrapper
@torch.fx.wrap
def fused_global_pool_view(input_tensor, view_shape):
    """Fuse adaptive_avg_pool2d to size 1 with view operation"""
    # Get input dimensions
    batch_size, channels, height, width = input_tensor.shape
    
    # Parse view shape
    if len(view_shape) == 2:
        view_dim0, view_dim1 = view_shape
    else:
        raise ValueError(f"Unsupported view shape: {view_shape}")
    
    # Create output tensor
    output = torch.zeros(batch_size, view_dim0, view_dim1, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Calculate grid dimensions - each program handles one output element
    output_size = batch_size * view_dim0 * view_dim1
    grid = (output_size,)
    
    # Launch Triton kernel
    fused_global_pool_view_kernel[grid](
        input_tensor,
        output,
        batch_size,
        channels, 
        height,
        width,
        view_dim0,
        view_dim1,
        BLOCK_SIZE=1024,  # Use 1024 for processing spatial blocks
    )
    
    return output

# Replacement function
def replacement_func():
    return fused_global_pool_view