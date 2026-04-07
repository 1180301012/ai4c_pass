import torch
import triton
import triton.language as tl

def pattern(query, key, value):
    # This is the attention computation pattern:
    # matmul = torch.matmul(query, key)
    # tmp_1 = matmul * 1.0
    # tmp_2 = torch.nn.functional.softmax(tmp_1, dim = -1, dtype = torch.float32)
    # tmp_3 = tmp_2.to(torch.float32)
    # tmp_4 = torch.nn.functional.dropout(tmp_3, p = 0.0, training = False)
    # to = tmp_4.to(value.dtype)
    # matmul_1 = torch.matmul(to, value)
    # tmp_6 = matmul_1.transpose(1, 2)
    # tmp_7 = tmp_6.contiguous()
    # tmp_8 = tmp_7.reshape(1, 257, -1)
    # tmp_9 = tmp_8.contiguous()
    
    matmul = torch.matmul(query, key) * 1.0
    attention_weights = torch.nn.functional.softmax(matmul, dim=-1, dtype=torch.float32)
    # Convert to float32 for dropout, then back to original dtype
    attention_weights = attention_weights.to(torch.float32)
    # Dropout with p=0.0 is no-op, so we skip it
    attention_weights = attention_weights.to(value.dtype)
    attention_output = torch.matmul(attention_weights, value)
    tmp_6 = attention_output.transpose(1, 2)
    tmp_7 = tmp_6.contiguous()
    tmp_8 = tmp_7.reshape(1, 257, -1)
    tmp_9 = tmp_8.contiguous()
    return tmp_9

def replacement_args(query, key, value):
    return (query, key, value)

@triton.jit
def softmax_kernel(
    x_ptr,
    out_ptr,
    batch_size, num_heads, seq_len, head_dim,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    # We process one row per program
    row_idx = tl.program_id(0)
    col_start = tl.program_id(1) * BLOCK_N
    
    # Calculate global positions
    batch = row_idx // (num_heads * seq_len)
    head = (row_idx // seq_len) % num_heads
    seq = row_idx % seq_len
    
    # Load row of logits
    x_cols = tl.arange(0, BLOCK_N)
    mask = x_cols < seq_len
    logits_ptr = x_ptr + batch * num_heads * seq_len * seq_len + head * seq_len * seq_len + seq * seq_len + x_cols
    logits = tl.load(logits_ptr, mask=mask, other=-float('inf'))
    
    # Compute softmax
    max_val = tl.max(logits, 0)
    shifted = logits - max_val
    exp = tl.exp(shifted)
    norm = tl.sum(exp, 0)
    out = exp / norm
    
    # Store result
    out_ptr = out_ptr + batch * num_heads * seq_len * seq_len + head * seq_len * seq_len + seq * seq_len + x_cols
    tl.store(out_ptr, out, mask=mask)

@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    batch_size, num_heads, m, n, k,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Calculate program position
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_batch = tl.program_id(2)
    
    # Compute ranges
    m_range = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_range = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Iterate over k dimension
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_offset in range(0, k, BLOCK_K):
        k_range = k_offset + tl.arange(0, BLOCK_K)
        
        # Load a and b tiles
        a_mask = m_range[:, None] < m and k_range[None, :] < k
        b_mask = k_range[:, None] < k and n_range[None, :] < n
        
        a_ptr_tile = a_ptr + pid_batch * m * k + m_range[:, None] * k + k_range[None, :]
        b_ptr_tile = b_ptr + pid_batch * k * n + k_range[:, None] * n + n_range[None, :]
        
        a_tile = tl.load(a_ptr_tile, mask=a_mask, other=0.0)
        b_tile = tl.load(b_ptr_tile, mask=b_mask, other=0.0)
        
        acc += tl.dot(a_tile, b_tile)
    
    # Store result
    c_ptr_tile = c_ptr + pid_batch * m * n + m_range[:, None] * n + n_range[None, :]
    mask = m_range[:, None] < m and n_range[None, :] < n
    tl.store(c_ptr_tile, acc, mask=mask)

@torch.fx.wrap
def optimized_attention(query, key, value):
    batch_size, num_heads, seq_len_q, head_dim = query.shape
    _, _, seq_len_k, _ = key.shape
    _, _, seq_len_v, _ = value.shape
    
    # Input shapes validation
    assert seq_len_q == seq_len_k == seq_len_v, "Sequences must have same length"
    assert head_dim == key.shape[-1], "Head dimension mismatch"
    assert seq_len_v * head_dim == 4096, "Final output dimension must be 4096"
    
    # Convert to float32 for numerical stability
    orig_dtype = query.dtype
    query_f32 = query.to(torch.float32)
    key_f32 = key.to(torch.float32)
    value_f32 = value.to(torch.float32)
    
    # Matmul: query @ key.transpose(-2, -1)
    attention_logits = torch.empty(batch_size, num_heads, seq_len_q, seq_len_k, dtype=torch.float32, device=query.device)
    
    if batch_size * num_heads * seq_len_q * seq_len_k > 0:
        matmul_kernel[
            (batch_size * num_heads * (seq_len_q + BLOCK_M - 1) // BLOCK_M, 
             (seq_len_k + BLOCK_N - 1) // BLOCK_N, 1),
            (BLOCK_M, BLOCK_N, BLOCK_K)
        ](
            query_f32,
            key_f32,
            attention_logits,
            batch_size, num_heads, seq_len_q, seq_len_k, head_dim,
            BLOCK_M=32, BLOCK_N=32, BLOCK_K=32
        )
    
    # Softmax along last dimension
    attention_weights = attention_logits.softmax(dim=-1)
    
    # Matmul: attention_weights @ value
    attention_output = torch.empty(batch_size, num_heads, seq_len_q, head_dim, dtype=torch.float32, device=query.device)
    
    if batch_size * num_heads * seq_len_q * head_dim > 0:
        matmul_kernel[
            (batch_size * num_heads * (seq_len_q + BLOCK_M - 1) // BLOCK_M, 
             (head_dim + BLOCK_N - 1) // BLOCK_N, 1),
            (BLOCK_M, BLOCK_N, BLOCK_K)
        ](
            attention_weights,
            value_f32,
            attention_output,
            batch_size, num_heads, seq_len_q, head_dim, seq_len_v,
            BLOCK_M=32, BLOCK_N=32, BLOCK_K=32
        )
    
    # Convert back to original dtype
    attention_output = attention_output.to(orig_dtype)
    
    # Reshape to output format
    # First transpose: [batch, heads, seq, dim] -> [batch, seq, heads, dim]
    output_transposed = attention_output.transpose(1, 2)
    
    # Then reshape: [batch, seq, heads, dim] -> [batch, seq, heads*dim] = [1, 257, 4096]
    output_reshaped = output_transposed.reshape(1, 257, -1)
    
    return output_reshaped

def replacement_func():
    return optimized_attention