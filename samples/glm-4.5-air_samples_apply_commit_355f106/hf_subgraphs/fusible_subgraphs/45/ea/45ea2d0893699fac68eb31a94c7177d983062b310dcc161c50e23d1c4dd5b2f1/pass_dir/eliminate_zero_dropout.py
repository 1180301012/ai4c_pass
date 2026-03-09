import torch
import triton
import triton.language as tl

# Simple pattern: a + b for matching addition operations
def pattern(a, b):
    return a + b

def replacement_args(a, b):
    return (a, b)

@triton.jit
def fused_linear_add_kernel(
    x_ptr, weight_ptr, bias_ptr, residual_ptr, out_ptr, intermediate_ptr,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # Row and column offsets for the program
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    
    if pid >= grid_m:
        return
    
    # Compute M and N offsets
    m_offset = pid * BLOCK_SIZE_M
    m_indices = m_offset + tl.arange(0, BLOCK_SIZE_M)
    k_indices = tl.arange(0, K)
    
    # Load bias - broadcast along the M dimension
    bias = tl.load(bias_ptr + k_indices, mask=k_indices < N, other=0.0)
    bias = bias[None, :]  # [1, N] for broadcasting
    
    # Load residual slice
    residual_slice = tl.load(residual_ptr + m_indices[:, None] * N + k_indices[None, :], 
                           mask=(m_indices[:, None] < M)[:, None] & (k_indices[None, :] < N), 
                           other=0.0).to(tl.float32)
    
    # Linear computation: x @ weight + bias
    acc = tl.zeros((BLOCK_SIZE_M, N), dtype=tl.float32)
    x_ptrs = x_ptr + m_indices[:, None] * K
    weight_ptrs = weight_ptr + k_indices[:, None] * N
    
    # Loop over K dimension for GEMM
    for k in range(0, K, BLOCK_SIZE_K):
        k_offset = k
        k_mask = k_offset + k_indices < K
        
        # Load x and weight tiles
        x_tile = tl.load(x_ptrs + k_offset, mask=(m_indices[:, None] < M) & k_mask[None, :], other=0.0).to(tl.float32)
        weight_tile = tl.load(weight_ptrs + k_offset * N, mask=k_mask[:, None] & (k_indices[None, :] < N), other=0.0).to(tl.float32)
        
        # Matrix multiplication
        acc += tl.dot(x_tile, weight_tile, out_dtype=tl.float32)
    
    # Add bias
    acc += bias
    
    # Add residual
    out = acc + residual_slice
    
    # Store results
    out_mask = (m_indices[:, None] < M) & (k_indices[None, :] < N)
    tl.store(out_ptr + m_indices[:, None] * N + k_indices[None, :], out, mask=out_mask)
    tl.store(intermediate_ptr + m_indices[:, None] * N + k_indices[None, :], acc, mask=out_mask)

@torch.fx.wrap
def simple_add(a, b):
    # Simple addition kernel for matched pattern a + b
    return a + b

def replacement_func():
    return simple_add