import torch
import triton
import triton.language as tl

# Pattern matching for linear + reshape fusion
def pattern(tmp_0, in_1):
    # Linear operation with variable assignment
    tmp_1 = torch.nn.functional.linear(in_1, tmp_0, None)
    # Reshape operation with exact dimensions
    tmp_2 = tmp_1.reshape(1, 197, 3, -1, 48)
    return tmp_2

# Extract arguments for replacement
def replacement_args(tmp_0, in_1):
    return (tmp_0, in_1)

# Optimized kernel combining linear + reshape
@triton.jit
def linear_reshape_kernel(
    weight_ptr, x_ptr, out_ptr,
    batch_size, seq_len, in_features, out_features, inner_dim,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # Program identifiers for parallel execution
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    
    # Compute current block ranges
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    k_offsets = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    m_mask = m_offsets < batch_size * seq_len
    
    # Load weight matrix
    weight = tl.load(weight_ptr + k_offsets[:, None] * out_features + m_offsets[None, :], 
                    mask=k_offsets[:, None] < out_features, other=0.0)
    
    # Load input matrix
    x = tl.load(x_ptr + m_offsets[:, None] * in_features + k_offsets[None, :], 
               mask=m_offsets[:, None] < batch_size * seq_len, other=0.0)
    
    # Compute matrix multiplication
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    for k in range(0, in_features, BLOCK_SIZE_N):
        # Load current block of x
        x_block = tl.load(x_ptr + m_offsets[:, None] * in_features + (k + tl.arange(0, BLOCK_SIZE_N))[None, :], 
                         mask=(m_offsets[:, None] < batch_size * seq_len) & (k + tl.arange(0, BLOCK_SIZE_N))[:, None] < in_features, other=0.0)
        # Load current block of weight
        weight_block = tl.load(weight_ptr + (k + tl.arange(0, BLOCK_SIZE_N))[:, None] * out_features + m_offsets[None, :], 
                              mask=(k + tl.arange(0, BLOCK_SIZE_N))[:, None] < out_features & (m_offsets[None, :] < batch_size * seq_len), other=0.0)
        # accumulate
        acc += tl.dot(x_block, weight_block, trans_b=True)
    
    # Reshape output directly to final target shape [1, 197, 3, inner_dim, 48]
    # Calculate target indices for the reshape
    m_idx = m_offsets // seq_len  # batch dimension (should be 0 for our case)
    s_idx = m_offsets % seq_len   # sequence dimension (0-196)
    
    # Map to final tensor layout: [1, 197, 3, inner_dim, 48]
    target_m = m_idx
    target_s = s_idx
    target_c = pid_k // (inner_dim * 48)  # channel dim (0-2)
    target_i = (pid_k % (inner_dim * 48)) // 48  # inner dim
    target_d = pid_k % 48  # depth dim
    
    # Calculate output offset
    out_offset = (target_m * 197 * 3 * inner_dim * 48 + 
                  target_s * 3 * inner_dim * 48 + 
                  target_c * inner_dim * 48 + 
                  target_i * 48 + 
                  target_d)
    
    # Store result
    tl.store(out_ptr + out_offset, acc, mask=m_mask)

@torch.fx.wrap
def linear_reshape_fused(weight, x):
    # Get tensor shapes
    batch_size = x.shape[0]
    seq_len = x.shape[1]
    in_features = x.shape[2]
    out_features = weight.shape[0]
    
    # Calculate inner dimension: (out_features // 3 // 48)
    inner_dim = out_features // (3 * 48)
    
    # Output shape: [1, 197, 3, inner_dim, 48]
    out_shape = (1, seq_len, 3, inner_dim, 48)
    out = torch.empty(out_shape, dtype=torch.float32, device=x.device)
    
    # Set block sizes based on tensor characteristics
    BLOCK_SIZE_M = 64  # Process multiple sequences together
    BLOCK_SIZE_K = 128  # Features dimension
    BLOCK_SIZE_N = 32   # Parallel compute width
    
    # Calculate grid dimensions
    m_dim = batch_size * seq_len
    k_dim = out_features  # This determines how many parallel computations we need
    
    # Launch kernel
    linear_reshape_kernel[(triton.cdiv(m_dim, BLOCK_SIZE_M), triton.cdiv(k_dim, BLOCK_SIZE_K))](
        weight_ptr=weight,
        x_ptr=x,
        out_ptr=out,
        batch_size=batch_size,
        seq_len=seq_len,
        in_features=in_features,
        out_features=out_features,
        inner_dim=inner_dim,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return out

def replacement_func():
    return linear_reshape_fused