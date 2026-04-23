"""
Optimize the EfficientNet classifier head pattern.

Pattern: adaptive_avg_pool2d(output_size=1) + flatten

This fusion computes per-channel mean directly instead of first reducing to 1x1 then flattening.
Uses a highly optimized Triton kernel with autotuning.
"""
import torch
import triton
import triton.language as tl


# Autotuned kernel for optimal performance
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=1, num_warps=4),
    ],
    key=['n_elements'],
)
@triton.jit
def adaptive_avg_pool_flatten_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    n_channels,
    h,
    w,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute per-channel mean pooling and flatten in a single kernel.
    
    Uses a 1D grid with block-level parallelization.
    Each block computes BLOCK_SIZE output elements cooperatively.
    """
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create masks for valid elements
    mask = offsets < n_elements
    
    # Calculate batch and channel indices for each thread
    batch_idx = offsets // n_channels
    channel_idx = offsets % n_channels
    
    # Base offset for each thread
    base_offsets = batch_idx * n_channels * h * w + channel_idx * h * w
    
    # Parallel reduction over spatial dimensions
    # Each thread handles one output element
    sum_vals = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    for h_idx in range(h):
        for w_idx in range(w):
            load_offsets = base_offsets + h_idx * w + w_idx
            vals = tl.load(input_ptr + load_offsets, mask=mask, other=0.0)
            sum_vals = sum_vals + vals
    
    # Compute mean
    mean_vals = sum_vals / (h * w)
    
    # Store results
    tl.store(output_ptr + offsets, mean_vals, mask=mask)


def adaptive_avg_pool_flatten_wrapper(input_tensor):
    """
    Wrapper for the fused adaptive_avg_pool2d + flatten operation.
    Input: [N, C, H, W] -> Output: [N, C]
    """
    n, c, h, w = input_tensor.shape
    output = torch.empty((n, c), dtype=input_tensor.dtype, device=input_tensor.device)
    
    n_elements = n * c
    # Use simple 1D grid
    grid = (n_elements,)
    
    adaptive_avg_pool_flatten_kernel[grid](
        input_ptr=input_tensor,
        output_ptr=output,
        n_elements=n_elements,
        n_channels=c,
        h=h,
        w=w,
    )
    
    return output


def pattern(x):
    """
    Pattern: adaptive_avg_pool2d + flatten(1)
    
    Matches the EfficientNet classifier head which computes:
    1. Adaptive average pooling to 1x1
    2. Flatten to 1D
    """
    pooled = torch.nn.functional.adaptive_avg_pool2d(x, 1)
    flat = torch.flatten(pooled, 1)
    return flat


def replacement_args(x):
    return (x,)


def replacement_func():
    return adaptive_avg_pool_flatten_wrapper