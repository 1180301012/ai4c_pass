import torch
import triton
import triton.language as tl

@triton.jit
def fused_matmul_transpose_kernel(
    attn_ptr,
    value_ptr,
    output_ptr,
    batch_size,
    seq_len,
    num_heads,
    head_dim,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused kernel for matmul followed by transpose optimization"""
    # Program IDs
    m_pid = tl.program_id(0)
    n_pid = tl.program_id(1)
    
    # Calculate ranges
    m_offset = m_pid * BLOCK_M
    n_offset = n_pid * BLOCK_N
    
    # Create shared memory for tiles
    attn_tile = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    value_tile = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
    output_tile = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Load tiles into shared memory
    for k_offset in range(0, num_heads, BLOCK_K):
        k_block = min(BLOCK_K, num_heads - k_offset)
        
        # Load attention tile: [BLOCK_M, BLOCK_K] 
        for m in range(BLOCK_M):
            for k in range(k_block):
                m_idx = m_offset + m
                k_idx = k_offset + k
                if m_idx < seq_len and k_idx < num_heads:
                    attn_val = tl.load(attn_ptr + m_idx * num_heads + k_idx)
                    attn_tile[m, k] = attn_val
        
        # Load value tile: [BLOCK_K, BLOCK_N]
        for k in range(k_block):
            for n in range(BLOCK_N):
                k_idx = k_offset + k
                n_idx = n_offset + n
                if k_idx < num_heads and n_idx < head_dim:
                    val_idx = k_idx * head_dim + n_idx
                    val_val = tl.load(value_ptr + val_idx)
                    value_tile[k, n] = val_val
        
        # Matrix multiplication: output = attn @ value
        for m in range(BLOCK_M):
            for n in range(BLOCK_N):
                acc = 0.0
                for k in range(k_block):
                    acc += attn_tile[m, k] * value_tile[k, n]
                m_idx = m_offset + m
                n_idx = n_offset + n
                if m_idx < seq_len and n_idx < head_dim:
                    # Transpose storage: store as [n_idx, m_idx] for output
                    output_idx = n_idx * seq_len + m_idx
                    prev_val = tl.load(output_ptr + output_idx, mask=False)
                    output_tile[m, n] = prev_val + acc
    
    # Store result back
    for m in range(BLOCK_M):
        for n in range(BLOCK_N):
            m_idx = m_offset + m
            n_idx = n_offset + n
            if m_idx < seq_len and n_idx < head_dim:
                # Transpose storage: store as [n_idx, m_idx] for output
                output_idx = n_idx * seq_len + m_idx
                output_val = output_tile[m, n] 
                tl.store(output_ptr + output_idx, output_val)

@torch.fx.wrap
def fused_matmul_transpose(attention, value):
    """Optimized fused matmul and transpose for attention computation"""
    batch_size, seq_len, num_heads = attention.shape
    value_batch_size, value_num_heads, head_dim = value.shape
    
    assert batch_size == value_batch_size
    assert num_heads == value_num_heads
    
    # Create output tensor directly in transposed layout: [batch, head_dim, seq_len]
    output_shape = (batch_size, head_dim, seq_len)
    output = torch.empty(output_shape, dtype=attention.dtype, device=attention.device)
    
    # Process each batch item separately (simulating the fusion)
    for b in range(batch_size):
        # Reset output for this batch
        batch_output = torch.zeros((head_dim, seq_len), dtype=attention.dtype, device=attention.device)
        
        # Call Triton kernel for this batch
        # Triton requires 1D/2D grids for simplicity
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        
        # Calculate grid dimensions
        m_dim = (seq_len + BLOCK_M - 1) // BLOCK_M
        n_dim = (head_dim + BLOCK_N - 1) // BLOCK_N
        
        fused_matmul_transpose_kernel[(m_dim, n_dim)](
            attn_ptr=attention[b, :, :],
            value_ptr=value[b, :, :],
            output_ptr=batch_output,
            batch_size=batch_size,
            seq_len=seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K
        )
        
        # Store in output tensor
        output[b, :, :] = batch_output
    
    return output

def pattern(in_0, in_1):
    """Pattern: matmul followed by transpose"""
    tmp_0 = torch.matmul(in_0, in_1)
    tmp_1 = tmp_0.permute(0, 2, 1)
    return (tmp_1,)

def replacement_args(in_0, in_1):
    """Extract arguments for the replacement function"""
    return (in_0, in_1)

def replacement_func():
    """Return the optimized fused matmul-transpose function"""
    return fused_matmul_transpose