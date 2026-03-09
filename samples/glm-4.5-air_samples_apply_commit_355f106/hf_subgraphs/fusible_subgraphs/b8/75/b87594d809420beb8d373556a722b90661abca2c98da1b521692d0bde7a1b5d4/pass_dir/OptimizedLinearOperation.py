import torch
import triton
import triton.language as tl

# Pattern matching function for linear operation (matmul + bias)
def pattern(input, weight, bias):
    # Match torch.nn.functional.linear(input, weight, bias)
    # input: [batch_size, in_features]
    # weight: [out_features, in_features] 
    # bias: [out_features]
    output = torch.nn.functional.linear(input, weight, bias)
    return output

# Argument extraction function
def replacement_args(input, weight, bias):
    return (input, weight, bias)

# Optimized Triton kernel for linear operation
@triton.jit
def linear_kernel(
    input_ptr, 
    weight_ptr, 
    bias_ptr, 
    output_ptr, 
    batch_size,
    in_features,
    out_features,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Each program handles one output element
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute bounds
    m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Compute output bounds
    m_mask = m < batch_size
    n_mask = n < out_features
    
    # Matrix multiplication
    for k in range(0, in_features):
        # Load input elements
        input_vals = tl.load(input_ptr + m[:, None] * in_features + k, mask=m_mask[:, None], other=0.0)
        
        # Load weights
        weight_vals = tl.load(weight_ptr + k * out_features + n[None, :], mask=n_mask[None, :], other=0.0)
        
        # Accumulate
        acc += input_vals * weight_vals
    
    # Add bias
    bias_vals = tl.load(bias_ptr + n[None, :], mask=n_mask[None, :], other=0.0)
    acc = acc + bias_vals
    
    # Store result
    tl.store(output_ptr + m[:, None] * out_features + n[None, :], acc, mask=m_mask[:, None] and n_mask[None, :])

# Kernel wrapper
@torch.fx.wrap
def optimized_linear(input, weight, bias):
    batch_size, in_features = input.shape
    out_features = weight.shape[0]
    
    # Determine block sizes - adjust based on our specific problem shapes
    BLOCK_SIZE_M = 8    # Block size for batch dimension (smaller for better utilization)
    BLOCK_SIZE_N = 32   # Block size for output features dimension
    
    # Calculate grid dimensions
    num_blocks_m = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_blocks_n = (out_features + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (num_blocks_m, num_blocks_n)
    
    # Create output tensor
    output = torch.empty((batch_size, out_features), device=input.device, dtype=input.dtype)
    
    # Launch kernel
    linear_kernel[grid](
        input,
        weight,
        bias,
        output,
        batch_size,
        in_features,
        out_features,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N
    )
    
    return output

# Replacement function
def replacement_func():
    return optimized_linear