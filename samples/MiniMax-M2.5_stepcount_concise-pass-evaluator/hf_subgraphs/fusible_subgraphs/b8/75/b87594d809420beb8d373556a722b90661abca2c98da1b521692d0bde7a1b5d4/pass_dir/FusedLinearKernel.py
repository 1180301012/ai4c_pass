import torch
import triton
import triton.language as tl


# Pattern matching function - matches the linear operation from the model
def pattern(in_2, weight, bias):
    """
    Match the linear operation: torch.nn.functional.linear(in_2, weight, bias)
    This is: y = x @ W^T + bias
    """
    result = torch.nn.functional.linear(in_2, weight, bias)
    return result


# Argument extraction function
def replacement_args(in_2, weight, bias):
    return (in_2, weight, bias)


# Optimized Triton kernel for linear layer - simplified version
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_stages=1, num_warps=1),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=1, num_warps=1),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=1, num_warps=1),
    ],
    key=['batch_size'],
)
@triton.jit
def triton_linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, out_features, in_features,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Simple Triton kernel for linear layer: y = x @ W^T + bias
    Each program processes one batch element and computes all output features.
    """
    # Get program ID (one per batch element)
    pid = tl.program_id(0)
    
    # Only process valid batch elements
    if pid >= batch_size:
        return
    
    # Offsets for output features
    offs_n = tl.arange(0, BLOCK_SIZE)
    mask_n = offs_n < out_features
    
    # Initialize accumulator for each output feature
    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Iterate over input features
    for k in range(in_features):
        # Load input element
        input_val = tl.load(input_ptr + pid * in_features + k)
        
        # Load weight column for all output features
        weight_ptrs = weight_ptr + k + offs_n * in_features
        weight_vals = tl.load(weight_ptrs, mask=mask_n, other=0.0)
        
        # Multiply and accumulate
        accumulator += input_val * weight_vals
    
    # Add bias if provided
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
        accumulator += bias
    
    # Store result
    output_ptrs = output_ptr + pid * out_features + offs_n
    tl.store(output_ptrs, accumulator, mask=mask_n)


@torch.fx.wrap
def triton_linear_kernel_wrapper(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to launch the Triton linear kernel.
    """
    batch_size = input.shape[0]
    out_features = weight.shape[0]
    in_features = weight.shape[1]
    
    # Allocate output
    output = torch.empty((batch_size, out_features), device=input.device, dtype=input.dtype)
    
    # Calculate grid - one thread per batch element
    grid = (batch_size,)
    
    # Launch kernel
    triton_linear_kernel[grid](
        input, weight, bias, output,
        batch_size, out_features, in_features,
    )
    
    return output


def replacement_func():
    return triton_linear_kernel_wrapper