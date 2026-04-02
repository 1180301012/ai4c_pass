import torch
import triton
import triton.language as tl

# Pattern matching for softmax + matmul + transpose sequence
def pattern(x, y):
    # This matches the sequence: softmax -> matmul -> transpose
    tmp_13 = x.softmax(dim = -1)
    matmul_1 = tmp_13 @ y
    tmp_15 = matmul_1.transpose(-1, -2)
    return tmp_15

# Extract arguments for the replacement function
def replacement_args(x, y):
    return (x, y)

# Optimized Triton kernel for softmax + matmul fusion
@triton.jit
def softmax_matmul_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    batch_size,
    m,
    n,
    k,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    batch = pid // ((m + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M * (n + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)
    m_block = (pid % ((m + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M * (n + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)) // ((n + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)
    n_block = (pid % ((m + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M * (n + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)) % ((n + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)
    
    if batch >= batch_size:
        return
    
    # Create accumulator matrix for this block
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float16)
    
    # Iterate over k dimension
    for k_idx in range(0, k, BLOCK_SIZE_K):
        k_end = min(k_idx + BLOCK_SIZE_K, k)
        
        # Load block of y matrix (transposed)
        y_block = tl.load(y_ptr + 
                         batch * k * n +
                         k_idx * n +
                         n_block * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N),
                         mask=n_block * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) < n,
                         other=0.0)
        y_block = y_block.to(tl.float16)
        
        # Process rows in this block
        for m_idx in range(m_block * BLOCK_SIZE_M, min((m_block + 1) * BLOCK_SIZE_M, m)):
            # Load row of x matrix and apply softmax
            x_row = tl.load(x_ptr +
                           batch * m * k +
                           m_idx * k +
                           tl.arange(k_idx, k_end),
                           mask=tl.arange(k_idx, k_end) < k,
                           other=0.0)
            
            # Compute softmax for this row
            max_val = tl.max(x_row)
            exp_x = tl.exp(x_row - max_val)
            sum_exp = tl.sum(exp_x)
            softmax_row = exp_x / sum_exp
            
            # Compute matmul contribution for this row
            for i in range(BLOCK_SIZE_N):
                if n_block * BLOCK_SIZE_N + i < n:
                    accumulator[m_idx - m_block * BLOCK_SIZE_M, :] += softmax_row * y_block[i]
    
    # Store the result to output matrix (transposed, so we store as n x m)
    m_idx_global = m_block * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_idx_global = n_block * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create 2D grid of indices
    m_indices = m_idx_global[:, None]
    n_indices = n_idx_global[None, :]
    
    # Store results (transposed output so we store n x m as m x n)
    for i in range(BLOCK_SIZE_M):
        for j in range(BLOCK_SIZE_N):
            if (m_block * BLOCK_SIZE_M + i < m and 
                n_block * BLOCK_SIZE_N + j < n and
                batch < batch_size):
                out_idx = batch * m * n + (m_block * BLOCK_SIZE_M + i) * n + (n_block * BLOCK_SIZE_N + j)
                value = accumulator[i, j]
                tl.store(out_ptr + out_idx, value)

# Wrapper function for the optimized kernel
@torch.fx.wrap
def fuse_softmax_matmul(x, y):
    batch_size, m, k = x.shape
    _, n, _ = y.shape
    
    # Output will be [batch_size, m, n] but we need to return transpose [batch_size, n, m]
    temp_output = torch.zeros(batch_size, m, n, dtype=x.dtype, device=x.device)
    final_output = torch.zeros(batch_size, n, m, dtype=x.dtype, device=x.device)
    
    # Calculate launch grid
    block_size_m = 32
    block_size_n = 32
    block_size_k = 32
    
    grid_m = (m + block_size_m - 1) // block_size_m
    grid_n = (n + block_size_n - 1) // block_size_n
    grid_size = batch_size * grid_m * grid_n
    
    # Launch the kernel
    softmax_matmul_kernel[grid_size](
        x_ptr=x,
        y_ptr=y,
        out_ptr=temp_output,
        batch_size=batch_size,
        m=m,
        n=n,
        k=k,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
    )
    
    # Perform transpose operation
    final_output = temp_output.transpose(-1, -2)
    
    return final_output

# Replacement function (must return a callable)
def replacement_func():
    return fuse_softmax_matmul