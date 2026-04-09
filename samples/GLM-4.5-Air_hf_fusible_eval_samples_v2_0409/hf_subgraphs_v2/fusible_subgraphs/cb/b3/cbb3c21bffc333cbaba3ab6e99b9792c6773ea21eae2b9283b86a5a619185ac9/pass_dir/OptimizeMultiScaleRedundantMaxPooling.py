import torch
import triton
import triton.language as tl

def pattern(in_0):
    # Pattern: ReLU followed by 3 identical max_pool2d operations + concatenation
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_2 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_3 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_4 = torch.cat([tmp_0, tmp_1, tmp_2, tmp_3], 1)
    return tmp_4

def replacement_args(in_0):
    return (in_0,)

# Efficient max pooling kernel with computed values reused multiple times
@triton.jit
def compute_once_reuse_max_pool_kernel(
    x_ptr,
    out_identity_ptr,
    out_pooled_ptr,  # This will be used for all 3 pooling operations since they're identical
    n_channels,
    height,
    width,
    kernel_size,
    stride,
    padding,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Efficient kernel that computes max_pool2d once and reuses the result for multiple scales.
    Since all 3 max_pool2d operations are identical, we compute once and reuse.
    """
    pid = tl.program_id(0)
    
    # Each thread processes a pixel in a channel
    channel = pid // (height * width)
    idx_in_channel = pid % (height * width)
    h = idx_in_channel // width
    w = idx_in_channel % width
    
    if channel >= n_channels or h >= height or w >= width:
        return
    
    # Get base pointer for the channel
    x_ch_ptr = x_ptr + channel * height * width
    
    # Load input value and store as identity
    x_val = tl.load(x_ch_ptr + idx_in_channel)
    tl.store(out_identity_ptr + channel * height * width + idx_in_channel, x_val)
    
    # Compute max pooling once
    max_val = x_val
    
    # Apply 5x5 max pooling with stride=1, padding=2
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            ph = h - padding + kh
            pw = w - padding + kw
            
            if 0 <= ph < height and 0 <= pw < width:
                p_idx = ph * width + pw
                val = tl.load(x_ch_ptr + p_idx)
                if val > max_val:
                    max_val = val
    
    # Store the pooled result - same result used for all 3 pooling operations
    tl.store(out_pooled_ptr + channel * height * width + idx_in_channel, max_val)

@torch.fx.wrap
def optimized_multi_scale_forward(in_0):
    """
    Optimized forward pass that eliminates redundant max_pool2d computations:
    1. Apply ReLU on input
    2. Compute max_pool2d ONCE and reuse result 3 times
    3. Concatenate identity + 3x identical max_pool results efficiently
    """
    # Get input dimensions
    batch_size, n_channels, height, width = in_0.shape
    device = in_0.device
    
    # Apply ReLU
    x_relu = torch.relu(in_0)
    
    # Create intermediate output tensors
    out_identity = torch.empty((batch_size, n_channels, height, width), 
                              device=device, dtype=x_relu.dtype)
    out_pooled = torch.empty((batch_size, n_channels, height, width), 
                            device=device, dtype=x_relu.dtype)
    
    # Launch optimized Triton kernel for each batch element
    kernel_size = 5
    stride = 1 
    padding = 2
    
    # Use BLOCK_SIZE optimized for GPU occupancy
    BLOCK_SIZE = 256  # Number of pixels processed per thread group
    
    total_elements = n_channels * height * width
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    for b in range(batch_size):
        # Call optimized kernel that computes identity and pooled result
        compute_once_reuse_max_pool_kernel[(num_programs,)](
            x_ptr=x_relu[b, :, :, :],
            out_identity_ptr=out_identity[b, :, :, :],
            out_pooled_ptr=out_pooled[b, :, :, :],
            n_channels=n_channels,
            height=height,
            width=width,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            BLOCK_SIZE=BLOCK_SIZE
        )
    
    # Efficient concatenation: identity + 3x identical pooled results
    # torch.cat([out_identity, out_pooled, out_pooled, out_pooled], dim=1)
    out = torch.cat([out_identity, out_pooled, out_pooled, out_pooled], dim=1)
    
    return out

def replacement_func():
    return optimized_multi_scale_forward