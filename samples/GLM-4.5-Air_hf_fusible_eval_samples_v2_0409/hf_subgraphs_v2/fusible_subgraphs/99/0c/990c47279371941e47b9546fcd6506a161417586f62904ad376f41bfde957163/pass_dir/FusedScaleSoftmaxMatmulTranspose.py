import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    # Match the original computation sequence
    tmp_0 = 0.0625 * in_0
    tmp_1 = torch.nn.functional.softmax(tmp_0, dim=-1)
    out = torch.matmul(tmp_1, in_1)
    result = out.permute(0, 2, 1)
    return result

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_kernel(
    # Input tensors
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    # Shape information
    batch_size,
    m_dim,      # 8192
    k_dim,      # 19  
    n_dim,      # 256
    # Constants
    scale: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    # Get program IDs for batch, block rows, and block columns
    batch_id = tl.program_id(0)
    block_m = tl.program_id(1) * BLOCK_M
    block_n = tl.program_id(2) * BLOCK_N
    
    # Create offsets within the batch
    m_offsets = block_m + tl.arange(0, BLOCK_M)
    n_offsets = block_n + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)
    
    # Create masks to handle boundary conditions
    m_mask = m_offsets < m_dim
    n_mask = n_offsets < n_dim
    k_mask = k_offsets < k_dim
    
    # Compute base pointers for this batch
    batch_in_0_base = in_0_ptr + batch_id * m_dim * k_dim
    batch_in_1_base = in_1_ptr + batch_id * k_dim * n_dim
    batch_out_base = out_ptr + batch_id * n_dim * m_dim
    
    # Initialize accumulator for GEMM
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)
    
    # Compute in stages along k dimension
    for k_pos in range(0, k_dim, BLOCK_K):
        # Calculate current k-block offsets and mask
        current_k_offsets = k_pos + k_offsets
        current_k_mask = current_k_offsets < k_dim
        
        # Load from in_0 (scaled) - stride: [batch, m, k]
        in_0_ptrs = batch_in_0_base[:, None] + m_offsets[:, None] * k_dim + current_k_offsets[None, :]
        in_0_val = tl.load(in_0_ptrs, mask=current_k_mask[None, :], other=0.0).to(tl.float32)
        
        # Scale the in_0 values
        in_0_scaled = in_0_val * scale
        
        # Compute softmax for this k-block
        # Get max values for stability
        max_vals = tl.maximum.reduce(in_0_scaled, axis=1)
        # Subtract max and exponentiate
        exp_vals = tl.exp(in_0_scaled - max_vals[:, None])
        # Sum over k dimension and normalize
        sum_exp = tl.sum(exp_vals, axis=1)
        softmax_vals = exp_vals / sum_exp[:, None]
        
        # Load from in_1 - stride: [batch, k, n]
        in_1_ptrs = batch_in_1_base[:, None] + current_k_offsets[:, None] * n_dim + n_offsets[None, :]
        in_1_val = tl.load(in_1_ptrs, mask=current_k_mask[:, None], other=0.0).to(tl.float32)
        
        # Update accumulator: A^softmax @ B
        softmax_expanded = softmax_vals[:, :, None]  # Add n dimension
        accumulator += tl.sum(softmax_expanded * in_1_val[None, :, :], axis=1)
    
    # Final transpose: accumulate has shape [BLOCK_M, BLOCK_N], need to write as [BLOCK_N, BLOCK_M]
    # Transpose by swapping dimensions
    accumulator_t = accumulator.trans()
    
    # Store the final result (transposed)
    out_ptrs = batch_out_base[:, None] + n_offsets[:, None] * m_dim + m_offsets[None, :]
    tl.store(out_ptrs, accumulator_t.to(tl.float16), mask=m_mask[:, None] & n_mask[None, :])

@triton.jit
def optimized_fused_kernel(
    in_0_ptr,
    in_1_ptr, 
    out_ptr,
    batch_size,
    m_dim,
    k_dim,
    n_dim,
    scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """More optimized version with better memory access patterns"""
    batch_id = tl.program_id(0)
    block_m = tl.program_id(1) * BLOCK_M
    block_n = tl.program_id(2) * BLOCK_N
    
    m_offsets = block_m + tl.arange(0, BLOCK_M)
    n_offsets = block_n + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)
    
    m_mask = m_offsets < m_dim
    n_mask = n_offsets < n_dim
    k_mask = k_offsets < k_dim
    
    # Base pointers
    in_0_batch_ptr = in_0_ptr + batch_id * m_dim * k_dim
    in_1_batch_ptr = in_1_ptr + batch_id * k_dim * n_dim
    out_batch_ptr = out_ptr + batch_id * n_dim * m_dim
    
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_pos in range(0, k_dim, BLOCK_K):
        current_k = k_pos + k_offsets
        k_mask = current_k < k_dim
        
        # Load in_0 blocks
        a_ptrs = in_0_batch_ptr[:, None] + m_offsets[:, None] * k_dim + current_k[None, :]
        a_val = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0).to(tl.float32)
        
        # Scale and softmax
        max_a = tl.maximum.reduce(a_val, axis=1)
        exp_a = tl.exp(a_val - max_a[:, None])
        sum_exp = tl.sum(exp_a, axis=1)
        softmax_a = exp_a / sum_exp[:, None]
        
        # Load in_1 blocks
        b_ptrs = in_1_batch_ptr[:, None] + current_k[:, None] * n_dim + n_offsets[None, :]
        b_val = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0).to(tl.float32)
        
        # GEMM: scale * A @ B, where A is softmax output
        accumulator += tl.dot(softmax_a, b_val)
    
    # Store transposed result
    out_ptrs = out_batch_ptr + (n_offsets[:, None] * m_dim + m_offsets[None, :])
    tl.store(out_ptrs, accumulator, mask=m_mask[:, None] & n_mask[None, :])

@triton.jit
def autotuned_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    batch_size,
    m_dim,
    k_dim,
    n_dim,
    scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr, 
    BLOCK_K: tl.constexpr,
):
    """Autotuned version with different configurations"""
    batch_id = tl.program_id(0)
    block_m = tl.program_id(1) * BLOCK_M
    block_n = tl.program_id(2) * BLOCK_N
    
    m_offsets = block_m + tl.arange(0, BLOCK_M)
    n_offsets = block_n + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)
    
    m_mask = m_offsets < m_dim
    n_mask = n_offsets < n_dim
    k_mask = k_offsets < k_dim
    
    in_0_base = in_0_ptr + batch_id * m_dim * k_dim
    in_1_base = in_1_ptr + batch_id * k_dim * n_dim
    out_base = out_ptr + batch_id * n_dim * m_dim
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_pos in range(0, k_dim, BLOCK_K):
        k = k_pos + k_offsets
        k_mask = k < k_dim
        
        # Load and scale in_0
        a_ptrs = in_0_base + m_offsets[:, None] * k_dim + k[None, :]
        a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0).to(tl.float32) * scale
        
        # Softmax along k dimension (columns in the loaded view)
        max_a = tl.maximum.reduce(a, axis=1)
        exp_a = tl.exp(a - max_a[:, None])
        sum_exp = tl.sum(exp_a, axis=1)
        softmax_a = exp_a / sum_exp[:, None]
        
        # Load in_1
        b_ptrs = in_1_base + k[:, None] * n_dim + n_offsets[None, :]
        b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0).to(tl.float32)
        
        # Matrix multiplication
        acc += tl.dot(softmax_a, b)
    
    # Store transposed result
    o_ptrs = out_base + n_offsets[:, None] * m_dim + m_offsets[None, :]
    tl.store(o_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])

@torch.fx.wrap
def fused_operation(in_0, in_1):
    # Get tensor properties
    batch_size, m_dim, k_dim = in_0.shape
    _, _, n_dim = in_1.shape
    
    # Determine optimal block sizes based on tensor dimensions
    if m_dim >= 1024:
        BLOCK_M = 64
    elif m_dim >= 512:
        BLOCK_M = 32  
    else:
        BLOCK_M = 16
        
    if n_dim >= 512:
        BLOCK_N = 64
    elif n_dim >= 256:
        BLOCK_N = 32
    else:
        BLOCK_N = 16
        
    # For small k_dim, use larger BLOCK_K, otherwise smaller
    if k_dim <= 32:
        BLOCK_K = min(k_dim, 32)
    else:
        BLOCK_K = 16
    
    # Calculate grid dimensions
    grid_m = (m_dim + BLOCK_M - 1) // BLOCK_M
    grid_n = (n_dim + BLOCK_N - 1) // BLOCK_N
    
    # Create output tensor with desired shape (transposed)
    out = torch.empty((batch_size, n_dim, m_dim), device=in_0.device, dtype=in_0.dtype)
    
    # Launch kernel
    grid = (batch_size, grid_m, grid_n)
    
    autotuned_kernel[grid](
        in_0,
        in_1,
        out,
        batch_size,
        m_dim,
        k_dim, 
        n_dim,
        0.0625,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K
    )
    
    return out

def replacement_func():
    return fused_operation