import torch
import triton
import triton.language as tl

def pattern(x, weight, bias):
    # Linear transformation
    linear_out = torch.nn.functional.linear(x, weight, bias)
    # Transpose operation (permute 0,2,1)
    transposed = linear_out.permute(0, 2, 1)
    # Reshape operation
    reshaped = transposed.reshape(x.shape[0], -1, 16, 16)
    return linear_out, transposed, reshaped

def replacement_args(x, weight, bias):
    return (x, weight, bias)

@triton.jit
def fused_linear_transpose_reshape_kernel(
    x_ptr, 
    weight_ptr, 
    bias_ptr,
    out_ptr,
    batch_size,
    seq_len,
    hidden_size,
    out_features,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    # Program ID - iterates over batches
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)      # Iterates over output features
    pid_n = tl.program_id(2)      # Iterates over sequence length
    
    # Create offset masks for bounds checking
    m_mask = pid_m < (out_features + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    n_mask = pid_n < (seq_len + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Calculate tile bounds
    m_offset = pid_m * BLOCK_SIZE_M
    n_offset = pid_n * BLOCK_SIZE_N
    
    # Allocate shared memory tiles
    x_tile = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    weight_tile = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
    bias_tile = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Compute matrix multiplication tile by tile
    for k in range((hidden_size + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K):
        # Load x tile (batch slice)
        k_offset = k * BLOCK_SIZE_K
        x_mask = (m_offset < out_features) & (k_offset + tl.arange(0, BLOCK_SIZE_K) < hidden_size)
        weight_mask = (k_offset + tl.arange(0, BLOCK_SIZE_K) < hidden_size) & (n_offset + tl.arange(0, BLOCK_SIZE_N) < seq_len)
        
        if m_mask:
            x_tile = tl.load(x_ptr + pid_batch * seq_len * hidden_size + 
                           n_offset * hidden_size + k_offset + 
                           tl.arange(0, BLOCK_SIZE_K).reshape(1, -1),
                           mask=x_mask, other=0.0)
        
        if k_mask := (k_offset + tl.arange(0, BLOCK_SIZE_K) < hidden_size):
            weight_tile = tl.load(weight_ptr + 
                                (m_offset + tl.arange(0, BLOCK_SIZE_M)).reshape(-1, 1) * hidden_size + 
                                k_offset + 
                                tl.arange(0, BLOCK_SIZE_K).reshape(1, -1),
                                mask=k_mask.reshape(-1, 1) & weight_mask.reshape(1, -1), other=0.0)
        
        if m_mask and n_mask:
            bias_tile = tl.load(bias_ptr + m_offset + tl.arange(0, BLOCK_SIZE_M),
                              mask=(m_offset + tl.arange(0, BLOCK_SIZE_M)) < out_features, other=0.0)
    
    # GEMM operation
    if m_mask and n_mask:
        output = tl.dot(x_tile, weight_tile.to(tl.float32))
        output = output + bias_tile.reshape(-1, 1)
        # Store result with bounds checking
        out_offset = pid_batch * seq_len * out_features + n_offset * out_features + m_offset
        tl.store(out_ptr + out_offset + 
                tl.arange(0, BLOCK_SIZE_M).reshape(-1, 1) + 
                tl.arange(0, BLOCK_SIZE_N).reshape(1, -1) * BLOCK_SIZE_M,
                output, mask=(m_offset + tl.arange(0, BLOCK_SIZE_M)) < out_features & 
                           (n_offset + tl.arange(0, BLOCK_SIZE_N)) < seq_len)

@torch.fx.wrap
def fused_linear_transpose_reshape(x, weight, bias):
    batch_size, seq_len, hidden_size = x.shape
    out_features = bias.shape[0]
    
    # Output size for linear layer: [batch_size, seq_len, out_features]
    linear_out = torch.empty((batch_size, seq_len, out_features), device=x.device, dtype=x.dtype)
    
    # Triton kernel configuration
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    
    # Calculate grid dimensions
    grid_m = (out_features + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (seq_len + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (batch_size, grid_m, grid_n)
    
    # Launch kernel
    fused_linear_transpose_reshape_kernel[grid, (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=linear_out,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        out_features=out_features,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    # Perform transpose and reshape
    transposed = linear_out.permute(0, 2, 1)  # [batch, out_features, seq_len]
    reshaped = transposed.reshape(batch_size, -1, 16, 16)
    
    return linear_out, transposed, reshaped

def replacement_func():
    return fused_linear_transpose_reshape