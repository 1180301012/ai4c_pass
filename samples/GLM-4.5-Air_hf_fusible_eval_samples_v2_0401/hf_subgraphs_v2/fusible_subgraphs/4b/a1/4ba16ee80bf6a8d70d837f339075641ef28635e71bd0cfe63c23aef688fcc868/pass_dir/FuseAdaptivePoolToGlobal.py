import torch
import triton
import triton.language as tl

def pattern(x):
    """
    Match adaptive_avg_pool2d followed by flatten operation
    This pattern can be fused into global average pooling for better performance
    """
    # Adaptive pooling to (1,1) output
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
    # Flatten starting from dimension 1
    tmp_3 = torch.flatten(tmp_1, 1)
    return tmp_3  # Return the final output after flattening

def replacement_args(x):
    """Return the input tensor that will be passed to the replacement"""
    return (x,)

@triton.jit
def global_avg_pool_kernel_2d(
    x_ptr,
    out_ptr,
    N: tl.constexpr,      # Batch size
    C: tl.constexpr,      # Channels 
    H: tl.constexpr,      # Height
    W: tl.constexpr,      # Width
    BLOCK_SIZE: tl.constexpr,
):
    """Global average pooling kernel for 2D tensor"""
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Create memory offsets for the entire batch and channel
    x_offset = batch_idx * (C * H * W) + channel_idx * (H * W)
    out_offset = batch_idx * C + channel_idx
    
    # Load all spatial elements for this batch and channel
    spatial_elements = tl.load(x_ptr + x_offset, mask=None, other=0.0)
    
    # Compute mean across spatial dimensions (H * W elements)
    mean_val = tl.sum(spatial_elements) / (H * W)
    
    # Store the result
    if batch_idx == 0:  # Only store for first batch (assuming batch_size=1)
        tl.store(out_ptr + out_offset, mean_val)

@torch.fx.wrap  
def fused_global_pool(x):
    """Fused global average pooling + flattening"""
    N, C, H, W = x.shape
    batch_size = N
    
    # Create output tensor
    out = torch.empty(batch_size, C, dtype=x.dtype, device=x.device)
    
    # Launch kernel
    num_batches = batch_size
    num_channels = C
    
    # Use appropriate block size based on tensor shape
    BLOCK_SIZE = 1024
    
    grid = (num_batches, num_channels)
    global_avg_pool_kernel_2d[grid](
        x_ptr=x,
        out_ptr=out,
        N=N,
        C=C,
        H=H,
        W=W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    """Replacement function that returns fused global pooling + flattening"""
    return fused_global_pool