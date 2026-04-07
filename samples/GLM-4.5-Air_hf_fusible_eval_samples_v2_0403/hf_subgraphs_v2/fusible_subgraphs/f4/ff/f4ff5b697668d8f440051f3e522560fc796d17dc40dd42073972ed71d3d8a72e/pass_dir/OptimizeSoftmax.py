import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    """
    Pattern matching for the softmax operation:
    input: [N, 17, 4096] tensor
    output: torch.nn.functional.softmax(input, dim=2)
    """
    return torch.nn.functional.softmax(input_tensor, dim=2)

def replacement_args(input_tensor):
    return (input_tensor,)

@triton.jit
def optimized_softmax_kernel(
    input_ptr,
    output_ptr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized softmax kernel that processes each (N, H) pair independently
    
    Shape: [N, H, D] where we apply softmax along the D dimension (dim=2)
    """
    # Program ID determines which (N, H) pair this program handles
    pid = tl.program_id(0)
    n = pid // H
    h = pid % H
    
    # Check bounds
    if n >= N or h >= H:
        return
    
    # Start offset for this (n, h) pair: n * H * D + h * D
    base_offset = (n * H + h) * D
    
    # Load the entire D-dimension vector for this (n, h) pair
    offsets = base_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (base_offset + D)
    
    x = tl.load(input_ptr + offsets, mask=mask, other=-float('inf'))
    
    # Apply softmax - compute max, subtract max, exp, sum, divide
    max_val = tl.max(x, mask=mask)
    exp_x = tl.exp(x - max_val, mask=mask)
    sum_exp_x = tl.sum(exp_x, mask=mask)
    softmax_output = exp_x / sum_exp_x
    
    # Store the result
    tl.store(output_ptr + offsets, softmax_output, mask=mask)

@torch.fx.wrap
def optimized_softmax(input_tensor):
    """
    Wrapper function to launch the optimized softmax kernel
    """
    # Get tensor shape - expects [N, H, D] 
    N = input_tensor.shape[0]
    H = input_tensor.shape[1] 
    D = input_tensor.shape[2]
    
    # Create output tensor
    output_tensor = torch.empty_like(input_tensor)
    
    # Block size - should be a power of 2 for optimal performance
    BLOCK_SIZE = 1024  # Process 1024 elements at a time
    
    # Calculate number of programs needed - one per (N, H) pair
    num_pairs = N * H
    
    # Launch kernel
    grid = (num_pairs,)
    optimized_softmax_kernel[grid](
        input_tensor,
        output_tensor,
        N,
        H,
        D,
        BLOCK_SIZE
    )
    
    return output_tensor

def replacement_func():
    return optimized_softmax