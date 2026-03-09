import torch
import triton
import triton.language as tl

# Pattern matching function - matches linear + add pattern
def pattern(in_0, in_1, in_2, in_3):
    # tmp_0 = in_0, tmp_1 = in_1
    tmp_0 = in_0
    tmp_1 = in_1
    # tmp_2 = torch.nn.functional.linear(in_3, tmp_1, tmp_0)
    tmp_2 = torch.nn.functional.linear(in_3, tmp_1, tmp_0)
    # tmp_1 = tmp_0 = None
    tmp_1 = tmp_0 = None
    # tmp_3 = in_2 + tmp_2
    tmp_3 = in_2 + tmp_2
    # tmp_2 = None
    tmp_2 = None
    # Return tmp_3 as intermediate result (matches original computation flow)
    return (tmp_3,)

# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_3, in_1, in_0, in_2)

# Optimized kernel that fuses linear + add operations
@triton.jit
def linear_add_kernel(
    x_ptr,           # Input [1000, 128] 
    w_ptr,           # Weights [128, 128]
    bias_ptr,        # Bias [128]
    y_ptr,           # Add input [1000, 128]
    out_ptr,         # Output [1000, 128]
    M,               # Batch size (1000)
    K,               # Input features (128)
    N,               # Output features (128)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    # Program IDs for 2D grid
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute ranges this program should handle
    m_start = pid_m * BLOCK_SIZE_M
    m_end = min(m_start + BLOCK_SIZE_M, M)
    m_offs = m_start + tl.arange(0, BLOCK_SIZE_M)
    
    n_start = pid_n * BLOCK_SIZE_N
    n_end = min(n_start + BLOCK_SIZE_N, N)
    n_offs = n_start + tl.arange(0, BLOCK_SIZE_N)
    
    # Mask for M dimension
    mask_m = m_offs < M
    
    # Accumulators for matrix multiplication
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        k_end = min(k + BLOCK_SIZE_K, K)
        
        # Load weights for current K range
        w = tl.load(w_ptr + k * N + n_offs[None, :], 
                   mask=n_offs[None, :] < N, 
                   other=0.0)
        
        # Load bias and expand
        bias = tl.load(bias_ptr + n_offs, 
                      mask=n_offs < N, 
                      other=0.0)
        bias = bias[None, :]
        
        # Load x data
        x_ptrs = x_ptr + m_offs[:, None] * K + k
        # Create proper mask for [BLOCK_SIZE_M, BLOCK_SIZE_K] = [64, 32]
        k_valid = tl.arange(0, BLOCK_SIZE_K) < (K - k)  # [32]
        mask_x = (mask_m[:, None] and k_valid[None, :])  # [64, 32]
        x = tl.load(x_ptrs, 
                   mask=mask_x, 
                   other=0.0)
        
        # Load y data for addition
        y_ptrs = y_ptr + m_offs[:, None] * N + n_offs
        y = tl.load(y_ptrs, 
                   mask=mask_m[:, None] and n_offs[None, :] < N, 
                   other=0.0)
        
        # Accumulate matrix multiplication (y = x @ w.T + bias)
        acc += tl.dot(x, w.to(tl.float32), out_layout="MM")
        acc += bias
    
    # Add the y input (this implements y + (x @ W.T + bias))
    acc += y
    
    # Store output
    out_ptrs = out_ptr + m_offs[:, None] * N + n_offs
    tl.store(out_ptrs, acc, mask=mask_m[:, None] and n_offs[None, :] < N)

@torch.fx.wrap
def fused_linear_add(linear_input, weight, bias, add_input):
    # Get tensor shapes
    M, K = linear_input.shape
    N = weight.shape[1]
    
    # Block sizes optimized for our problem size [1000, 128]
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    
    # Calculate grid size
    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (grid_m, grid_n)
    
    # Allocate output
    output = torch.empty((M, N), dtype=torch.float32, device=linear_input.device)
    
    # Launch kernel
    linear_add_kernel[grid](
        linear_input,
        weight,
        bias,
        add_input,
        output,
        M, K, N,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    return output

# Replacement function
def replacement_func():
    return fused_linear_add