import torch
import triton
import triton.language as tl

def pattern(in_0):
    """Pattern to match SiLU followed by MaxPool2D operations"""
    tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
    tmp_1 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    return (tmp_1, tmp_0)

def replacement_args(in_0):
    """Extract arguments for the replacement function"""
    return (in_0,)

@triton.jit
def fused_silu_maxpool_kernel(
    input_ptr,
    silu_output_ptr, 
    maxpool_output_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Fused SiLU + MaxPool2D kernel"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute spatial coordinates
    h_start = pid_m * BLOCK_SIZE_M
    w_start = pid_n * BLOCK_SIZE_N
    
    # Create offsets for the input tensor
    h_offsets = h_start + tl.arange(0, BLOCK_SIZE_M)
    w_offsets = w_start + tl.arange(0, BLOCK_SIZE_N)
    h_mask = h_offsets < height
    w_mask = w_offsets < width
    
    # Process all channels in one go
    for c in range(0, channels, 1):
        # Load input patch (5x5 window)
        input_patch = tl.load(input_ptr + 
                             c * height * width +
                             h_offsets[:, None] * width + 
                             w_offsets, 
                             mask=h_mask[:, None] & w_mask,
                             other=-float('inf'))
        
        # Apply SiLU activation to the patch
        silu_patch = input_patch / (1.0 + tl.exp(-input_patch))
        
        # Store SiLU output
        tl.store(silu_output_ptr +
                 c * height * width +
                 h_offsets[:, None] * width + 
                 w_offsets,
                 silu_patch,
                 mask=h_mask[:, None] & w_mask)
        
        # Perform max pooling on the SiLU patch
        max_vals = tl.max(silu_patch, axis=1)
        max_indices = tl.argmax(silu_patch, axis=1)
        
        # Store max pooling result
        tl.store(maxpool_output_ptr +
                 c * height * width +
                 h_offsets + \
                 w_start,
                 max_vals,
                 mask=h_mask)

@torch.fx.wrap
def fused_silu_maxpool(input_tensor):
    """Wrapper function for the fused SiLU+MaxPool2D operation"""
    batch_size, channels, height, width = input_tensor.shape
    
    # Calculate optimal block sizes
    BLOCK_SIZE_H = 16  # Process 16 rows at a time
    BLOCK_SIZE_W = 16  # Process 16 columns at a time
    
    # Calculate grid dimensions
    grid_h = (height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_w = (width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    grid = (grid_h, grid_w)
    
    # Create output tensors
    silu_output = torch.empty_like(input_tensor)
    maxpool_output = torch.empty_like(input_tensor)
    
    # Launch kernel - process one batch at a time for simplicity
    for batch_idx in range(batch_size):
        fused_silu_maxpool_kernel[grid](
            input_ptr=input_tensor[batch_idx],
            silu_output_ptr=silu_output[batch_idx],
            maxpool_output_ptr=maxpool_output[batch_idx],
            batch_size=batch_size,
            channels=channels,
            height=height,
            width=width,
            BLOCK_SIZE_M=BLOCK_SIZE_H,
            BLOCK_SIZE_N=BLOCK_SIZE_W,
        )
    
    return (maxpool_output, silu_output)

def replacement_func():
    """Return the fused kernel function"""
    return fused_silu_maxpool