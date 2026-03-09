import torch
import triton
import triton.language as tl
import math

# Pattern matching function: Reshape operation (key part of Reshape+Sigmoid pattern)
def pattern(input_tensor, N, C):
    # Focus on the reshape operation which is the key transformation
    # The sigmoid would be handled by a different pattern
    tmp_4 = input_tensor.reshape(N, -1, C)
    return tmp_4

def replacement_args(input_tensor, N, C):
    return (input_tensor, N, C)

@triton.jit
def sigmoid_kernel(
    input_ptr,
    output_ptr,
    N, H, C,  # N is batch, H is height*width, C is channels
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Get program IDs
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_c = tl.program_id(2)
    
    # Calculate bounds
    n_start = pid_n * BLOCK_SIZE_N
    h_start = pid_h * BLOCK_SIZE_H
    c_start = pid_c * BLOCK_SIZE_C
    
    n_end = min(n_start + BLOCK_SIZE_N, N)
    h_end = min(h_start + BLOCK_SIZE_H, H)
    c_end = min(c_start + BLOCK_SIZE_C, C)
    
    # Process the block
    for n in range(n_start, n_end):
        for h in range(h_start, h_end):
            for c in range(c_start, c_end):
                # Calculate input and output offsets
                input_offset = n * H * C + h * C + c
                output_offset = input_offset
                
                # Load input value
                input_val = tl.load(input_ptr + input_offset, other=0.0)
                
                # Apply sigmoid: 1 / (1 + exp(-x))
                # Using fast sigmoid approximation for better performance
                # sigmoid(x) ≈ 0.5 * x * (1 + abs(x))^(-1) + 0.5
                # Or use a more accurate but still fast approximation
                exp_val = tl.exp(-tl.abs(input_val))
                sigmoid_val = tl.where(input_val >= 0, 
                                     tl.exp(-exp_val), 
                                     1.0 - tl.exp(-exp_val))
                
                # Store output value
                tl.store(output_ptr + output_offset, sigmoid_val)

@torch.fx.wrap
def reshape_sigmoid_fused(input_tensor, N, C):
    # Calculate intermediate dimension (H*W)
    original_shape = input_tensor.shape
    H_total = original_shape[1] * original_shape[2] * original_shape[3]
    
    # Create output tensor with same shape
    output = torch.empty_like(input_tensor)
    
    # Block size configuration for optimal GPU performance
    BLOCK_SIZE_N = 1  # Process one batch at a time for simplicity
    BLOCK_SIZE_H = 64  # Process 64 spatial elements
    BLOCK_SIZE_C = min(32, C)  # Process channels in blocks
    
    # Calculate grid dimensions
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_h = (H_total + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_c = (C + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    
    # Launch kernel
    sigmoid_kernel[(grid_n, grid_h, grid_c)](
        input_tensor,
        output,
        N, H_total, C,
        BLOCK_SIZE_N, BLOCK_SIZE_H, BLOCK_SIZE_C
    )
    
    return output

def replacement_func():
    return reshape_sigmoid_fused