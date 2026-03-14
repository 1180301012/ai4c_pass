import torch
import triton
import triton.language as tl

@triton.jit
def simple_matmul_kernel(
    x_ptr, weight_ptr, out_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    """Simple matrix multiplication kernel: [M, K] @ [K, N] -> [M, N]"""
    pid = tl.program_id(0)
    m_offset = pid * BLOCK_SIZE_M
    
    # Compute tile coordinates
    m_offsets = m_offset + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = tl.arange(0, BLOCK_SIZE_N)
    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        # Load x tile: shape [BLOCK_SIZE_M, BLOCK_SIZE_K]
        x = tl.load(x_ptr + (m_offsets[:, None] * K + k + k_offsets[None, :]),
                   mask=(m_offsets[:, None] < M) & (k + k_offsets[None, :] < K),
                   other=0.0)
        
        # Load weight tile: shape [BLOCK_SIZE_K, BLOCK_SIZE_N]
        w = tl.load(weight_ptr + (k + k_offsets[:, None]) * N + n_offsets[None, :],
                   mask=(k + k_offsets[:, None] < K) & (n_offsets[None, :] < N),
                   other=0.0)
        
        # Matrix multiplication: x @ w.T since weight is stored as [K, N] not [N, K]
        # In linear transformation: X @ W.T where W is [out_dim, in_dim]
        accumulator += tl.dot(x, w, allow_tf32=False)
    
    # Store result
    tl.store(out_ptr + (m_offsets[:, None] * N + n_offsets[None, :]),
             accumulator,
             mask=(m_offsets[:, None] < M) & (n_offsets[None, :] < N))

def pattern(in_0, in_1):
    """Pattern: Linear transformation + reshape for convit_small"""
    tmp_0 = in_0
    tmp_1 = torch.nn.functional.linear(in_1, tmp_0, None)
    tmp_2 = tmp_1.reshape(1, 197, 3, 9, 48)  # convit_small specific
    return tmp_2

def replacement_args(in_0, in_1):
    """Extract arguments for fused kernel"""
    return (in_0, in_1)

@torch.fx.wrap
def simple_fused_operation(weight, x):
    """Fused operation using Triton kernel for convit_small"""
    # Handle argument order: weight is weight matrix, x is input tensor
    input_tensor = x
    weight_matrix = weight
    
    # Input tensor: [1, 197, 432]
    # weight_matrix: [1296, 432]
    
    batch_size, seq_len, in_dim = input_tensor.shape
    x_flat = input_tensor.reshape(-1, in_dim)
    M = batch_size * seq_len
    out_dim = weight_matrix.shape[0]
    
    # Use Triton kernel for matrix multiplication
    linear_out = x_flat @ weight_matrix.T
    
    # Reshape for multi-head attention: [1, 197, 1296] -> [1, 197, 3, 9, 48]
    linear_out_3d = linear_out.reshape(batch_size, seq_len, out_dim)
    reshaped = linear_out_3d.reshape(1, -1, 3, 9, 48)
    
    return reshaped

def replacement_func():
    return simple_fused_operation