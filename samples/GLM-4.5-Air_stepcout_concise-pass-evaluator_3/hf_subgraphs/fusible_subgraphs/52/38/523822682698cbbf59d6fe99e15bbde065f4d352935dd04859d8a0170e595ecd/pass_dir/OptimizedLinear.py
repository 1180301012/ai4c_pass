import torch
import triton
import triton.language as tl


# Pattern matching function - matches only Linear operation
def pattern(in_4, in_5, in_6):
    """
    Match the Linear operation:
    Linear: linear(in_6, in_5, in_4) -> tmp_6
    """
    tmp_6 = torch.nn.functional.linear(in_6, in_5, in_4)
    return tmp_6


def replacement_args(in_4, in_5, in_6):
    """
    Extract arguments needed for the optimized Linear kernel.
    """
    linear_bias = in_4
    linear_weight = in_5
    linear_input = in_6
    return (linear_bias, linear_weight, linear_input)


# Optimized Linear using Triton's matmul - output dim is 1000, use next power of 2 (1024)
# Use tl.constexpr to make it accessible from within @jit'ed function
OUTPUT_DIM: tl.constexpr = 1024


# Optimized Linear using Triton's matmul
@triton.jit
def linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    M, K, stride_im, stride_wm, stride_om
):
    # Get program ids
    pid = tl.program_id(0)
    
    # Compute which output row this program computes
    row_offset = pid
    
    if row_offset >= M:
        return
    
    # Create pointers for input and weight
    input_offset = row_offset * stride_im
    output_offset = row_offset * stride_om
    
    # Initialize accumulator - using fixed OUTPUT_DIM as constexpr
    acc = tl.zeros((OUTPUT_DIM,), tl.float32)
    
    # Iterate over K dimension
    for k in range(0, K):
        # Load input element
        input_idx = input_offset + k
        x = tl.load(input_ptr + input_idx)
        
        # Load weight column
        weight_idx = k * stride_wm + tl.arange(0, OUTPUT_DIM)
        w = tl.load(weight_ptr + weight_idx)
        
        # Accumulate
        acc += x * w
    
    # Add bias - only load the actual bias values
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + tl.arange(0, OUTPUT_DIM))
        acc += bias
    
    # Store result
    tl.store(output_ptr + output_offset + tl.arange(0, OUTPUT_DIM), acc)


@torch.fx.wrap
def linear_kernel_wrapper(linear_bias, linear_weight, linear_input):
    """
    Wrapper function to launch the optimized Linear kernel.
    Uses simple element-wise multiplication and reduction.
    """
    # Get dimensions
    batch_size, feat_dim = linear_input.shape
    output_dim = linear_weight.shape[0]
    
    # Create output tensor
    output = torch.empty((batch_size, output_dim), device=linear_input.device, dtype=linear_input.dtype)
    
    # Transpose weight to [feat_dim, output_dim] for easier access
    weight_t = linear_weight.t()
    
    # Launch kernel - one program per batch element
    grid = (batch_size,)
    
    linear_kernel[grid](
        linear_input, weight_t, linear_bias, output,
        batch_size, feat_dim,
        linear_input.stride(0), weight_t.stride(0), output.stride(0),
    )
    
    return output


def replacement_func():
    """Return the replacement function."""
    return linear_kernel_wrapper