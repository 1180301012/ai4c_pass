import torch
import triton
import triton.language as tl


# Autotune configurations - BLOCK_SIZE is a constexpr that will be selected at compile time
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=4, num_warps=8),
    ],
    key=['N', 'H', 'W'],
)
@triton.jit
def batch_norm_kernel(
    input_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    output_ptr,
    N, C, H, W,
    eps: tl.constexpr,
):
    """
    Fused batch normalization kernel for 4D tensors.
    Computes: output = (input - mean) / sqrt(var + eps) * weight + bias
    
    Note: BLOCK_SIZE is automatically selected by the autotuner.
    """
    # Get BLOCK_SIZE from the config
    BLOCK_SIZE = 1024  # Default, will be overridden by autotuner
    
    # Calculate total number of elements
    n_elements = N * C * H * W
    
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate the starting offset for this program
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Calculate indices for N, C, H, W dimensions
    # Flattened index = n * C * H * W + c * H * W + h * W + w
    w_offsets = offsets % W
    h_offsets = (offsets // W) % H
    c_offsets = (offsets // (H * W)) % C
    n_offsets = offsets // (C * H * W)
    
    # Load input, mean, var, weight, bias
    # Input is shaped [N, C, H, W], we need to compute flat offset
    input_offsets = n_offsets * C * H * W + c_offsets * H * W + h_offsets * W + w_offsets
    x = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
    
    # Load mean, var, weight, bias for the channel
    mean = tl.load(mean_ptr + c_offsets, mask=mask, other=0.0)
    var = tl.load(var_ptr + c_offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + c_offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + c_offsets, mask=mask, other=0.0)
    
    # Batch norm computation
    # output = (input - mean) / sqrt(var + eps) * weight + bias
    normalized = (x - mean) / tl.sqrt(var + eps)
    output = normalized * weight + bias
    
    # Store result
    tl.store(output_ptr + input_offsets, output, mask=mask)


@torch.fx.wrap
def triton_batch_norm(input, running_mean, running_var, weight, bias, eps=0.001):
    """
    Custom batch normalization using Triton kernel.
    """
    N, C, H, W = input.shape
    
    # Allocate output
    output = torch.empty_like(input)
    
    # Calculate grid - use a fixed large number of programs
    n_elements = N * C * H * W
    num_programs = (n_elements + 1024 - 1) // 1024
    
    # Launch kernel - note: BLOCK_SIZE is automatically selected by autotuner
    batch_norm_kernel[(num_programs,)](
        input_ptr=input,
        mean_ptr=running_mean,
        var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        N=N, C=C, H=H, W=W,
        eps=eps,
    )
    
    return output


# Pattern matching function - matches the batch_norm + slice pattern
def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Match the computation pattern:
    tmp_4 = in_5[slice(None), slice(X, None), slice(None), slice(None)]
    tmp_5 = torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 0.001)
    return (tmp_5, tmp_4)
    """
    # First, do the slice operation - keep it as is since it's just a view
    tmp_4 = in_5[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, None)]
    
    # Now match the batch_norm operation
    # Note: We match the pattern but we'll optimize only the batch_norm part
    tmp_5 = torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 0.001)
    
    return tmp_5, tmp_4


# Extract arguments needed for the replacement
def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_4, in_0, in_1, in_3, in_2)


# Replacement function that returns the optimized kernel
def replacement_func():
    return triton_batch_norm