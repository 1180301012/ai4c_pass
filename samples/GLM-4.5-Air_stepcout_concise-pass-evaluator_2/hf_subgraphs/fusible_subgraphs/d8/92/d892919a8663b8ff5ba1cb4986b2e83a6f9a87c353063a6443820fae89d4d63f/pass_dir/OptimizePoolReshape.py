import torch
import triton
import triton.language as tl

def adaptive_avg_pool2d_view(x, output_size):
    """Pattern matching: Adaptive avg pool2d with output_size=1 followed by reshape"""
    # Pattern represents the structure: adaptive_avg_pool2d followed by reshape
    # We use placeholders to represent the operations without calling them
    pooled = x  # Placeholder for adaptive_avg_pool2d result
    if pooled.dim() == 4:  # [N,C,1,1]
        if pooled.shape[0] == 1:
            reshaped = pooled.view(1, pooled.shape[1])  # [1, C]
        else:
            reshaped = pooled.view(pooled.shape[0], pooled.shape[1])  # [N, C]
    else:
        reshaped = pooled
    return pooled, reshaped

def replacement_args(x, output_size):
    """Extract arguments for the optimized kernel"""
    return (x, output_size)

@triton.jit
def optimized_pooling_kernel(
    x_ptr,
    out_ptr,
    N,
    C,
    H,
    W,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    """Optimized kernel: direct mean computation without intermediate pooling"""
    # Compute grid coordinates
    n_idx = tl.program_id(0)
    c_idx = tl.program_id(1)
    
    # Ensure we don't go out of bounds
    n_mask = n_idx < N
    c_mask = c_idx < C
    
    # Create offset for current batch and channel
    batch_start = n_idx * C * H * W
    channel_start = c_idx * H * W
    
    # Compute mean over spatial dimensions directly
    spatial_elements = H * W
    sum_val = 0.0
    
    # Process spatial tiles
    spatial_offset = 0
    while spatial_offset < spatial_elements:
        offset = batch_start + channel_start + spatial_offset
        if offset < N * C * H * W:
            val = tl.load(x_ptr + offset, other=0.0)
            sum_val += val
            spatial_offset += 1
        else:
            spatial_offset = spatial_elements
    
    # Calculate mean
    mean_val = sum_val / spatial_elements
    
    # Store result at [n_idx, c_idx] position
    output_offset = n_idx * C + c_idx
    if n_mask and c_mask:
        tl.store(out_ptr + output_offset, mean_val)

@torch.fx.wrap
def optimized_adaptive_pool_resample(x, output_size=(1, 1)):
    """Optimized function: direct mean computation + reshape"""
    # Only handle 4D tensors and 1x1 output size for now
    if x.dim() != 4 or output_size != (1, 1):
        raise NotImplementedError("Only 4D tensors with 1x1 output size are supported")
    
    N, C, H, W = x.shape
    
    # Create output tensor
    if N == 1:
        out_shape = (1, C)
    else:
        out_shape = (N, C)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Block sizes for better memory access pattern
    BLOCK_SIZE_N = 1
    BLOCK_SIZE_C = 128  # Match with typical channel sizes in the graphs
    
    # Adjust block sizes if needed
    if C > 128:
        BLOCK_SIZE_C = 64
    if C > 256:
        BLOCK_SIZE_C = 32
    
    num_blocks_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_blocks_c = (C + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    
    # Launch kernel
    optimized_pooling_kernel[
        (num_blocks_n, num_blocks_c)
    ](
        x_ptr=x,
        out_ptr=out,
        N=N,
        C=C,
        H=H,
        W=W,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
    )
    
    return out, out

def replacement_func():
    """Return the optimized function"""
    return optimized_adaptive_pool_resample