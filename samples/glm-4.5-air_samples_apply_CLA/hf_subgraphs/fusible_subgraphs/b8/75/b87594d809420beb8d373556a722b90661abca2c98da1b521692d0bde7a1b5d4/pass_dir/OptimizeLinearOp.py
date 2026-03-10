import torch
import triton
import triton.language as tl

# Pattern matching function - match just the linear operation
def pattern(x, weight, bias):
    # Match just the linear operation pattern
    return torch.nn.functional.linear(x, weight, bias)

# Argument extraction function  
def replacement_args(bias, weight, input_tensor, sequence_output):
    # Extract arguments needed for the linear operation pattern
    # Note: All 4 parameters are accepted even if sequence_output is not used
    return (input_tensor, weight, bias)

# Optimized linear kernel
@triton.jit
def linear_kernel(
    x_ptr,           # input matrix [M, K]
    w_ptr,           # weight matrix [N, K] 
    b_ptr,           # bias vector [N]
    y_ptr,           # output matrix [M, N]
    M, N, K,         # dimensions
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    # Program identifiers for grid computation
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    
    # Compute ranges for this block
    m_start = pid_m * BLOCK_SIZE_M
    m_end = min((pid_m + 1) * BLOCK_SIZE_M, M)
    n_start = pid_n * BLOCK_SIZE_N  
    n_end = min((pid_n + 1) * BLOCK_SIZE_N, N)
    k_start = pid_k * BLOCK_SIZE_K
    k_end = min((pid_k + 1) * BLOCK_SIZE_K, K)
    
    # Initialize accumulator for this block
    acc = tl.zeros((m_end - m_start, n_end - n_start), dtype=tl.float32)
    
    # Load bias for this block of N
    bias = tl.load(b_ptr + tl.arange(n_start, n_end), mask=(tl.arange(n_start, n_end) < N))
    bias = bias[None, :]  # broadcast to [1, N]
    
    # Compute partial matrix-matrix products
    for k in range(k_start, k_end):
        # Load input slice [M_block, K_block] and weight slice [N_block, K_block]
        x = tl.load(x_ptr + m_start * K + k * M + tl.arange(0, m_end - m_start)[:, None],
                   mask=(tl.arange(m_start, m_end)[:, None] < M)[:, None] & (k < K))
        w = tl.load(w_ptr + n_start * K + k * N + tl.arange(0, n_end - n_start)[None, :],
                   mask=(tl.arange(n_start, n_end)[None, :] < N)[:, None] & (k < K))
        
        # Multiply and accumulate
        acc += x @ w.T
    
    # Add bias
    acc += bias
    
    # Store result
    y_offset = m_start * N + n_start
    tl.store(y_ptr + y_offset + tl.arange(m_end - m_start)[:, None] * N + tl.arange(n_end - n_start)[None, :],
             acc, mask=(tl.arange(m_start, m_end)[:, None] < M)[:, None] & (tl.arange(n_start, n_end)[None, :] < N))

@torch.fx.wrap
def triton_linear(bias, weight, input_tensor):
    M, K = input_tensor.shape
    N, _ = weight.shape
    
    # autotune configuration
    grid = lambda meta: (
        (M + meta['BLOCK_SIZE_M'] - 1) // meta['BLOCK_SIZE_M'],
        (N + meta['BLOCK_SIZE_N'] - 1) // meta['BLOCK_SIZE_N'],
        (K + meta['BLOCK_SIZE_K'] - 1) // meta['BLOCK_SIZE_K']
    )
    
    output = torch.empty((M, N), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Use efficient block sizes
    linear_kernel[grid](
        input_tensor,
        weight, 
        bias,
        output,
        M, N, K,
        BLOCK_SIZE_M=64,  # Process 64 rows at a time
        BLOCK_SIZE_N=256, # Process 256 columns at a time
        BLOCK_SIZE_K=32   # Process 32 elements of K dimension at a time
    )
    
    return output

def replacement_func():
    return triton_linear