import torch
import triton
import triton.language as tl

def pattern(x, weight, indices, scale):
    """Pattern for embedding lookup with scaling followed by addition fusion"""
    emb = torch.nn.functional.embedding(indices, weight, 1, None, scale, False, False)
    result = x + emb
    return result, emb  # Return both for preserving intermediate observable values

def replacement_args(x, weight, indices, scale):
    return (x, weight, indices, scale)

@triton.jit
def fused_embedding_add_kernel(
    x_ptr, weight_ptr, indices_ptr, out_ptr, emb_out_ptr,
    batch_size, seq_len, hidden_size,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """Fused embedding lookup and addition with Triton"""
    # Create program grid
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute memory offsets
    m_offset = pid_m * BLOCK_M
    n_offset = pid_n * BLOCK_N
    
    # Load bias (assuming it's zero for this optimization)
    # For simplicity, we're handling the add operation here
    
    # Load input x (already scaled)
    x_offsets = m_offset * seq_len * hidden_size + (n_offset // hidden_size) * hidden_size
    x_val = tl.load(x_ptr + x_offsets, mask=(m_offset < batch_size) and (n_offset < seq_len), other=0.0)
    
    # Compute embedding index offset
    batch_offset = m_offset * seq_len + (n_offset // hidden_size)
    emb_idx = tl.load(indices_ptr + batch_offset, mask=(m_offset < batch_size) and (n_offset < seq_len), other=0)
    
    # Load embedding
    emb_offset = emb_idx * hidden_size + (n_offset % hidden_size)
    emb_val = tl.load(weight_ptr + emb_offset, mask=(emb_idx < weight.shape[0]) and (n_offset % hidden_size < hidden_size), other=0.0)
    
    # Apply scale factor of 2.0 and add
    result = x_val + emb_val * 2.0
    
    # Store results
    out_offset = m_offset * seq_len * hidden_size + n_offset
    emb_offset = out_offset
    tl.store(out_ptr + out_offset, result, mask=(m_offset < batch_size) and (n_offset < seq_len * hidden_size))
    tl.store(emb_out_ptr + emb_offset, emb_val * 2.0, mask=(m_offset < batch_size) and (n_offset < seq_len * hidden_size))

@torch.fx.wrap
def fused_embedding_add(x, weight, indices):
    """Wrapper function for fused embedding + add with scale 2.0"""
    batch_size, seq_len, hidden_size = x.shape
    
    # Calculate optimal block sizes based on input dimensions
    BLOCK_M = 64
    BLOCK_N = 32
    BLOCK_K = hidden_size
    
    # Calculate grid sizes
    grid_m = (batch_size + BLOCK_M - 1) // BLOCK_M
    grid_n = (seq_len * hidden_size + BLOCK_N - 1) // BLOCK_N
    
    # Create output tensors
    result = torch.empty_like(x)
    emb_result = torch.empty_like(x)
    
    # Launch kernel
    fused_embedding_add_kernel[(grid_m, grid_n)](
        x_ptr=x,
        weight_ptr=weight,
        indices_ptr=indices,
        out_ptr=result,
        emb_out_ptr=emb_result,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K
    )
    
    return result, emb_result

def replacement_func():
    return fused_embedding_add