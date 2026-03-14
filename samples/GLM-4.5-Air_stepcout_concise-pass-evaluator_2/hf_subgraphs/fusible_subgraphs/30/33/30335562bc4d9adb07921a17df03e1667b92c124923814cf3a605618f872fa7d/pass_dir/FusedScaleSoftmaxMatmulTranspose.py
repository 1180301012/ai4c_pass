import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1):
    """
    Match the computation sequence:
    1. Scalar multiplication: 0.0625 * in_0
    2. Softmax along last dimension
    3. Matrix multiplication with in_1
    4. Transpose with permute(0, 2, 1)
    """
    tmp_0 = 0.0625 * in_0
    tmp_1 = torch.nn.functional.softmax(tmp_0, dim=-1)
    tmp_2 = torch.matmul(tmp_1, in_1)
    tmp_3 = tmp_2.permute(0, 2, 1)
    return tmp_3

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_kernel(
    x_ptr,           # Pointer to input tensor in_0 [B, M, N]
    y_ptr,           # Pointer to input tensor in_1 [B, N, K]
    out_ptr,         # Pointer to output tensor [B, K, M]
    batch_size,      # Batch dimension
    m_dim,           # M dimension (8192)
    n_dim,           # N dimension (19)
    k_dim,           # K dimension (256)
    scale_factor: tl.constexpr,  # 0.0625
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Compute program indices
    pid_b = tl.program_id(0)  # Batch index
    pid_m = tl.program_id(1)  # M dimension index
    
    # Create pointers for current batch
    x_batch_ptr = x_ptr + pid_b * m_dim * n_dim
    y_batch_ptr = y_ptr + pid_b * n_dim * k_dim
    out_batch_ptr = out_ptr + pid_b * k_dim * m_dim
    
    # Create offsets and masks
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = tl.arange(0, BLOCK_SIZE_N)
    
    m_mask = m_offsets < m_dim
    n_mask = n_offsets < n_dim
    
    # Initialize accumulator for matrix multiplication
    accumulator = tl.zeros((BLOCK_SIZE_M, k_dim), dtype=tl.float32)
    
    # Apply softmax along N dimension and compute matrix multiplication
    for n_start in range(0, n_dim, BLOCK_SIZE_N):
        n_local_offsets = n_start + n_offsets
        n_valid_mask = n_local_offsets < n_dim
        
        # Load X segment [BLOCK_SIZE_M, BLOCK_SIZE_N]
        x_ptr_addrs = x_batch_ptr + (m_offsets[:, None] * n_dim + n_local_offsets[None, :])
        x_values = tl.load(x_ptr_addrs, mask=m_mask[:, None] & n_valid_mask[None, :], other=0.0)
        
        # Apply scaling and softmax along last dimension (N)
        x_scaled = x_values * scale_factor
        max_val = tl.maximum.reduce(x_scaled, axis=1)
        x_normalized = x_scaled - max_val[:, None]
        exp_x = tl.exp(x_normalized)
        sum_exp = tl.sum(exp_x, axis=1)
        softmax_output = exp_x / sum_exp[:, None]
        
        # Load Y segment [BLOCK_SIZE_N, K] - we need to load the entire K dimension
        y_ptr_addrs = y_batch_ptr + (n_local_offsets[:, None] * k_dim + tl.arange(0, k_dim)[None, :])
        y_values = tl.load(y_ptr_addrs, mask=n_valid_mask[:, None] & (tl.arange(0, k_dim)[None, :] < k_dim), other=0.0)
        
        # Accumulate matrix multiplication result
        accumulator += tl.dot(softmax_output, y_values)
    
    # Store result directly to [B, K, M] layout by transposing the accumulator
    out_k_offsets = tl.arange(0, k_dim)
    out_m_offsets = m_offsets
    
    out_ptr_addrs = out_batch_ptr + (out_k_offsets[None, :] * m_dim + out_m_offsets[:, None])
    out_mask = (out_m_offsets[:, None] < m_dim) & (out_k_offsets[None, :] < k_dim)
    tl.store(out_ptr_addrs, accumulator, mask=out_mask)

@torch.fx.wrap
def fused_scale_softmax_matmul_transpose(in_0, in_1):
    """
    Fused kernel that combines:
    1. Scalar multiplication (0.0625)
    2. Softmax along last dimension
    3. Matrix multiplication with transposed result
    """
    # Get input shapes
    batch_size, m_dim, n_dim = in_0.shape
    _, _, k_dim = in_1.shape
    
    # Create output tensor with desired transpose shape [B, K, M]
    output = torch.empty((batch_size, k_dim, m_dim), dtype=in_0.dtype, device=in_0.device)
    
    # Calculate optimal block sizes - adjusted for our kernel structure
    BLOCK_SIZE_M = 128  # Larger block size for better GPU occupancy
    BLOCK_SIZE_N = 16   # Small block for N dimension (19 is small)
    
    # Calculate grid dimensions - only batch and M dimensions
    num_batch = batch_size
    num_m = (m_dim + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    
    # Launch kernel with 2D grid
    fused_kernel[(num_batch, num_m)](
        x_ptr=in_0,
        y_ptr=in_1,
        out_ptr=output,
        batch_size=batch_size,
        m_dim=m_dim,
        n_dim=n_dim,
        k_dim=k_dim,
        scale_factor=0.0625,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_scale_softmax_matmul_transpose