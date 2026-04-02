import torch
from torch import device
import triton
import triton.language as tl

def pattern(in_4, in_1, in_0, in_3, in_2):
    tmp_9 = in_4.unsqueeze(0)
    tmp_10 = tmp_9 + 2
    tmp_11 = torch.nn.functional.embedding(tmp_10, in_1, None, None, 2.0, False, False)
    tmp_12 = tmp_11.to(device(type='cuda', index=0))
    tmp_13 = in_0 + tmp_12
    tmp_14 = torch.nn.functional.layer_norm(tmp_13, (tmp_13.shape[-1],), in_3, in_2, 1e-05)
    return tmp_14

def replacement_args(in_4, in_1, in_0, in_3, in_2):
    return (in_4, in_1, in_0, in_3, in_2)

# Optimized fused embedding + addition + layer norm kernel
@triton.jit
def fused_embedding_add_norm_kernel(
    positions_ptr,
    embed_weights_ptr,
    input_ptr,
    gamma_ptr,
    beta_ptr,
    output_ptr,
    batch_size,
    seq_len,
    embed_dim,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Matrix dimensions
    M = batch_size * seq_len
    N = embed_dim
    K = embed_weights_ptr.shape[0]  # vocab_size
    
    # Program ID for matrix multiplication
    pid = tl.program_id(0)
    m_start = pid * BLOCK_SIZE_M
    m_end = min((pid + 1) * BLOCK_SIZE_M, M)
    
    # Offset for output
    offs_m = m_start * N + tl.arange(0, BLOCK_SIZE_N)
    mask = m_start < M
    
    # Load gamma and bias (layer norm parameters)
    gamma = tl.load(gamma_ptr + tl.arange(0, BLOCK_SIZE_N), mask=tl.arange(0, BLOCK_SIZE_N) < N, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + tl.arange(0, BLOCK_SIZE_N), mask=tl.arange(0, BLOCK_SIZE_N) < N, other=0.0).to(tl.float32)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    
    # Process each K dimension
    for k in range(0, K, 32):
        k_end = min(k + 32, K)
        
        # Load positions + 2 (embedding indices)
        pos_idx = tl.load(positions_ptr + (m_start // seq_len), mask=True)
        pos_idx = pos_idx + 2  # Add 2 as in original
        pos_idx = tl.max(0, tl.min(pos_idx, K - 1))  # Clamp to valid range
        
        # Load embedding weights for current K chunk
        weights = tl.load(embed_weights_ptr + pos_idx * N + tl.arange(0, BLOCK_SIZE_N), 
                         mask=tl.arange(0, BLOCK_SIZE_N) < N, other=0.0).to(tl.float32)
        
        # Load input
        input_val = tl.load(input_ptr + offs_m, mask=offs_m < M * N, other=0.0).to(tl.float32)
        
        # Matrix multiplication: input * embedding_weights
        accumulator += input_val * weights
    
    # Apply layer normalization
    mean = tl.sum(accumulator, axis=0) / N
    var = tl.sum((accumulator - mean) * (accumulator - mean), axis=0) / N
    std = tl.sqrt(var + 1e-05)
    normalized = (accumulator - mean) / std
    result = normalized * gamma + beta
    
    # Store output
    tl.store(output_ptr + offs_m, result, mask=mask)

@torch.fx.wrap
def fused_embedding_add_norm(positions, embed_weights, input_tensor, gamma, beta):
    batch_size, seq_len, embed_dim = input_tensor.shape
    
    # Handle cases where we still need to do embedding lookup
    if embed_weights.dim() == 2 and embed_weights.shape[1] == embed_dim:
        # Create output tensor
        output = torch.empty_like(input_tensor)
        
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 32
        num_programs = (batch_size * seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
        
        # Launch kernel
        fused_embedding_add_norm_kernel[(num_programs,)](
            positions_ptr=positions,
            embed_weights_ptr=embed_weights,
            input_ptr=input_tensor,
            gamma_ptr=gamma,
            beta_ptr=beta,
            output_ptr=output,
            batch_size=batch_size,
            seq_len=seq_len,
            embed_dim=embed_dim,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )
        
        return output
    else:
        # Fallback to original implementation if shapes don't match
        return pattern(positions, embed_weights, input_tensor, gamma, beta)

def replacement_func():
    return fused_embedding_add_norm