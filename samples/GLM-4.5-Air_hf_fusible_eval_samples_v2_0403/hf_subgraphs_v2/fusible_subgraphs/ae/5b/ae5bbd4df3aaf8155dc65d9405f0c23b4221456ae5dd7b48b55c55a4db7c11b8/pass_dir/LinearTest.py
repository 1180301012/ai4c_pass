import torch
import triton
import triton.language as tl

def pattern(in_3, in_1, in_0):
    """Simple linear transformation test pattern"""
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    return linear

def replacement_args(in_3, in_1, in_0):
    return (in_3, in_1, in_0)

@triton.jit
def linear_kernel(
    in_3_ptr, in_1_ptr, in_0_ptr, out_ptr,
    batch_size, seq_len, seq_len_2, hidden_dim, output_dim,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """Full linear transformation kernel"""
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute ranges for matrix multiplication
    m_range = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_range = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    k_range = tl.arange(0, BLOCK_SIZE_K)
    
    # Mask out of bounds
    m_mask = m_range < (batch_size * seq_len * seq_len_2)
    n_mask = n_range < output_dim
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over k dimension
    for k in range(0, hidden_dim, BLOCK_SIZE_K):
        k_block = k + k_range
        k_mask = k_block < hidden_dim
        
        # Load input tiles
        in_3_addr = (m_range[:, None] * hidden_dim + k_block[None, :]).to(tl.int32)
        in_3_tile = tl.load(in_3_ptr + in_3_addr, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        
        # Load weight tiles  
        weight_addr = (n_range[:, None] * hidden_dim + k_block[None, :]).to(tl.int32)
        weight_tile = tl.load(in_1_ptr + weight_addr, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        
        # Matrix multiplication
        accumulator += tl.dot(in_3_tile.to(tl.float32), weight_tile.to(tl.float32).T)
    
    # Load bias and add
    bias_addr = n_range.to(tl.int32)
    bias_tile = tl.load(in_0_ptr + bias_addr, mask=n_mask)
    
    # Add bias and store result
    out_addr = (m_range[:, None] * output_dim + n_range[None, :]).to(tl.int32)
    result = accumulator + bias_tile[None, :]
    tl.store(out_ptr + out_addr, result.to(tl.float16), mask=m_mask[:, None] & n_mask[None, :])

@torch.fx.wrap
def linear_test(in_3, in_1, in_0):
    """Wrapper for linear test"""
    # Calculate dimensions
    batch_size = in_3.shape[0]
    seq_len = in_3.shape[1] 
    seq_len_2 = in_3.shape[2]  # This is the time dimension (199)
    hidden_dim = in_3.shape[3]
    output_dim = in_1.shape[0]  # This is 8
    
    # Output shape: [batch_size, seq_len, seq_len_2, output_dim]
    out_shape = (batch_size * seq_len * seq_len_2, output_dim)
    out = torch.empty(out_shape, dtype=torch.float16, device=in_3.device)
    
    # Set up kernel configuration - must meet Triton's requirements (all dims >= 16)
    BLOCK_SIZE_M = 128  # Block size for M dimension 
    BLOCK_SIZE_N = 16   # Block size for N dimension (must be >= 16)
    BLOCK_SIZE_K = 32   # Block size for K dimension
    
    # Calculate grid dimensions
    m_dim = (batch_size * seq_len * seq_len_2 + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    n_dim = (output_dim + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel
    linear_kernel[(m_dim, n_dim)](
        in_3_ptr=in_3,
        in_1_ptr=in_1,
        in_0_ptr=in_0,
        out_ptr=out,
        batch_size=batch_size,
        seq_len=seq_len,
        seq_len_2=seq_len_2,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out

def replacement_func():
    return linear_test