import torch
import triton
from triton import autotune
import triton.language as tl


# Slice start position
SLICE_START = 2048


@autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_C': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_C': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_C': 512}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_C': 1024}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_C': 2048}, num_stages=4, num_warps=8),
    ],
    key=['C'],
)
@triton.jit
def batch_norm_kernel_autotuned(
    input_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    output_ptr, 
    N, C, H, W,
    eps: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Grid: (N, H, W) with C as inner loop
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    
    # Channel block iterator
    c_offsets = tl.arange(0, BLOCK_SIZE_C)
    
    # Load mean, var, weight, bias for all channels
    mean = tl.load(mean_ptr + c_offsets, mask=BLOCK_SIZE_C, other=0.0)
    var = tl.load(var_ptr + c_offsets, mask=BLOCK_SIZE_C, other=0.0)
    weight = tl.load(weight_ptr + c_offsets, mask=BLOCK_SIZE_C, other=1.0)
    bias = tl.load(bias_ptr + c_offsets, mask=BLOCK_SIZE_C, other=0.0)
    
    # Compute normalized std
    std = tl.sqrt(var + eps)
    
    # Normalize and affine transform
    norm = weight / std
    output_bias = bias - norm * mean
    
    # Process all channels
    for c_base in range(0, C, BLOCK_SIZE_C):
        c_offs = c_base + c_offsets
        c_mask = c_offs < C
        
        # Compute the linear index for input
        idx = ((pid_n * C + c_offs) * H + pid_h) * W + pid_w
        
        # Load input
        x = tl.load(input_ptr + idx, mask=c_mask, other=0.0)
        
        # Batch norm: (x - mean) / std * weight + bias
        out = x * norm + output_bias
        
        # Store output
        tl.store(output_ptr + idx, out, mask=c_mask)


@torch.fx.wrap
def triton_batch_norm(input, mean, var, weight, bias, training, momentum, eps):
    N, C, H, W = input.shape
    
    # Allocate output
    output = torch.empty_like(input)
    
    # Grid: (N, H, W) - each program handles one (n, h, w) position across all C
    grid = (N, H, W)
    
    batch_norm_kernel_autotuned[grid](
        input, mean, var, weight, bias,
        output,
        N, C, H, W,
        eps,
        BLOCK_SIZE_C=1024,  # Default, autotune will override
    )
    
    return output


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Match the batch_norm + slice pattern from the model.
    batch_norm is the main compute-intensive operation.
    """
    tmp_4 = in_5[(slice(None, None, None), slice(SLICE_START, None, None), slice(None, None, None), slice(None, None, None))]
    tmp_5 = torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 0.001)
    return (tmp_5, tmp_4)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_4, in_0, in_1, in_3, in_2, in_5, SLICE_START)


def replacement_func():
    return triton_batch_norm_fused


@torch.fx.wrap
def triton_batch_norm_fused(x, mean, var, weight, bias, in_5, slice_start):
    """
    Fused batch norm + slice operation.
    Applies batch norm to x and slices in_5.
    """
    # Apply batch norm
    bn_output = triton_batch_norm(x, mean, var, weight, bias, False, 0.1, 0.001)
    
    # Slice in_5: [slice(None), slice(slice_start, None), slice(None), slice(None)]
    sliced = in_5[:, slice_start:, :, :]
    
    return (bn_output, sliced)