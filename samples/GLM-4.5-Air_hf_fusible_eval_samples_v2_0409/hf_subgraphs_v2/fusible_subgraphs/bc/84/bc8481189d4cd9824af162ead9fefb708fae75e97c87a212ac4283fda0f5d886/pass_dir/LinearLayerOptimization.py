import torch
import triton
import triton.language as tl

# Pattern matching function for linear operation
def pattern(x, weight, bias):
    """Match linear operation: out = x @ weight.t() + bias"""
    result = torch.nn.functional.linear(x, weight, bias)
    return result

# Argument extraction function
def replacement_args(x, weight, bias):
    return (x, weight, bias)

# Triton kernel for optimized linear operation
@triton.jit
def linear_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    n_features, n_output, batch_size,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    """Optimized linear kernel using Triton"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute ranges
    m_range = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_range = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    m_mask = m_range < batch_size
    n_mask = n_range < n_output
    
    # Load bias vector
    bias = tl.load(bias_ptr + n_range, mask=n_mask, other=0.0)
    
    # Initialize output for this block
    acc = tl.zeros(bias.shape, dtype=tl.float32)
    
    # Loop over K dimension (features)
    for k in range(0, n_features, BLOCK_SIZE_K):
        k_range = k + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_range < n_features
        
        # Load x block and weight block
        x_block = tl.load(x_ptr + m_range[:, None] * n_features + k_range[None, :], 
                         mask=m_mask[:, None] and k_mask[None, :], other=0.0).to(tl.float32)
        
        weight_block = tl.load(weight_ptr + k_range[:, None] * n_output + n_range[None, :], 
                              mask=k_mask[:, None] and n_mask[None, :], other=0.0).to(tl.float32)
        
        # Matrix multiplication: m × k @ k × n = m × n
        acc += tl.dot(x_block, weight_block, acc_type=tl.float32)
    
    # Add bias and write to output
    out = acc + bias
    tl.store(out_ptr + m_range[:, None] * n_output + n_range[None, :], 
             out, mask=m_mask[:, None] and n_mask[None, :])

# Kernel wrapper
@torch.fx.wrap
def optimized_linear(x, weight, bias):
    """Optimized linear operation using Triton"""
    x_shape = x.shape
    batch_size = x_shape[0] if len(x_shape) > 1 else 1
    if len(x_shape) == 1:
        x = x[None, :]  # Add batch dimension
    
    n_features = x.shape[-1]
    n_output = weight.shape[0]
    
    # Set block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 64
    
    # Compute grid size
    num_blocks_m = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_blocks_n = (n_output + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Allocate output
    output = torch.empty(batch_size, n_output, dtype=x.dtype, device=x.device)
    
    # Launch kernel
    linear_kernel[(num_blocks_m, num_blocks_n)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=output,
        n_features=n_features,
        n_output=n_output,
        batch_size=batch_size,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    # Remove batch dimension if needed
    if len(x_shape) == 1:
        return output[0]
    return output

# Replacement function
def replacement_func():
    return optimized_linear