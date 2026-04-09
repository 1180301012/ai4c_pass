import torch
import triton
import triton.language as tl

def pattern(attention_scores, value_tensor):
    # Pattern matches softmax + matmul fusion in attention computation
    tmp_13 = attention_scores.softmax(dim = -1)
    matmul_1 = tmp_13 @ value_tensor
    out = matmul_1
    return out

def replacement_args(attention_scores, value_tensor):
    return (attention_scores, value_tensor)

@triton.jit
def fused_attention_kernel(
    scores_ptr,
    value_ptr,
    output_ptr,
    batch_size,
    seq_len,
    embed_dim,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused softmax + matmul attention kernel using Triton"""
    pid = tl.program_id(0)
    
    # Determine which program handles this output element
    total_elements = batch_size * seq_len * embed_dim
    element_idx = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = element_idx < total_elements
    
    if not mask[0]:
        return
    
    # Convert linear index to tensor coordinates
    batch_idx = element_idx // (seq_len * embed_dim) % batch_size
    seq_idx = element_idx // embed_dim % seq_len
    embed_idx = element_idx % embed_dim
    
    # Load the entire attention scores for this batch and sequence position
    scores_offset = batch_idx * seq_len * seq_len + seq_idx * seq_len
    
    # Load scores (this is the attention score vector for this query)
    scores = tl.load(scores_ptr + scores_offset + tl.arange(0, seq_len), 
                     mask=tl.arange(0, seq_len) < seq_len)
    
    # Compute softmax
    max_score = tl.max(scores)
    exp_scores = tl.exp(scores - max_score)
    sum_exp = tl.sum(exp_scores)
    softmax_weights = exp_scores / sum_exp
    
    # Load value matrix columns for this attention distribution
    # This is a simplified version - in practice we'd need more complex indexing
    # For now, assume we can load the required value elements
    
    # Compute weighted sum of values
    result = tl.zeros(1, dtype=tl.float32)
    valid_range = tl.arange(0, seq_len) < seq_len
    
    for k in range(0, seq_len, BLOCK_K):
        k_batch = tl.arange(k, k + BLOCK_K) < seq_len
        value_offset = batch_idx * seq_len * embed_dim + k * embed_dim + embed_idx
        
        # Load value elements
        values = tl.load(value_ptr + value_offset, 
                        mask=k_batch & (tl.arange(0, BLOCK_K) < seq_len - k))
        
        # Accumulate dot product
        weights = tl.load(softmax_weights + tl.arange(k, k + BLOCK_K), 
                         mask=k_batch & (tl.arange(0, BLOCK_K) < seq_len - k))
        result += tl.sum(values * weights)
    
    # Store result
    tl.store(output_ptr + element_idx, result, mask=mask)

@triton.jit
def fused_attention_kernel_optimized(
    scores_ptr,
    value_ptr,
    output_ptr,
    batch_size,
    seq_len_q,
    seq_len_k,
    embed_dim,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """More efficient fused attention kernel"""
    pid = tl.program_id(0)
    
    # Grid dimension: batch * seq_len_q * embed_dim
    total_elements = batch_size * seq_len_q * embed_dim
    element_idx = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = element_idx < total_elements
    
    if not mask[0]:
        return
    
    # Map to output tensor coordinates
    batch_idx = element_idx // (seq_len_q * embed_dim) % batch_size
    query_idx = element_idx // embed_dim % seq_len_q
    embed_idx = element_idx % embed_dim
    
    # Load attention scores for this query (shape: [seq_len_k])
    scores_offset = batch_idx * seq_len_q * seq_len_k + query_idx * seq_len_k
    scores = tl.load(scores_ptr + scores_offset + tl.arange(0, seq_len_k), 
                     mask=tl.arange(0, seq_len_k) < seq_len_k)
    
    # Compute softmax
    max_score = tl.max(scores)
    shifted_scores = scores - max_score
    exp_scores = tl.exp(shifted_scores)
    sum_exp = tl.sum(exp_scores)
    softmax_weights = exp_scores / sum_exp
    
    # Compute output = softmax_weights @ value_matrix[batch_idx, :, embed_idx]
    # We need to compute: sum_{key} softmax_weights[key] * value[batch_idx, key, embed_idx]
    
    # Vectorized reduction over keys
    result = tl.zeros(1, dtype=tl.float32)
    
    for k in range(0, seq_len_k, BLOCK_K):
        # Load attention weights
        k_offset = tl.arange(k, k + BLOCK_K) < seq_len_k
        weights = tl.load(softmax_weights + k, mask=k_offset)
        
        # Load values for this batch and embedding dimension
        value_offset = batch_idx * seq_len_k * embed_dim + k * embed_dim + embed_idx
        values = tl.load(value_ptr + value_offset, mask=k_offset)
        
        # Accumulate
        result += tl.sum(weights * values)
    
    # Store result
    output_offset = batch_idx * seq_len_q * embed_dim + query_idx * embed_dim + embed_idx
    tl.store(output_ptr + output_offset, result, mask=mask)

@torch.fx.wrap
def optimized_fused_attention(attention_scores, value_tensor):
    batch_size, seq_len_q, seq_len_k = attention_scores.shape
    embed_dim = value_tensor.shape[-1]
    
    # Create output tensor
    output = torch.empty(batch_size, seq_len_q, embed_dim, 
                        dtype=attention_scores.dtype, 
                        device=attention_scores.device)
    
    # Launch Triton kernel
    total_elements = batch_size * seq_len_q * embed_dim
    BLOCK_SIZE = 256
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_attention_kernel_optimized[(num_programs,)](
        scores_ptr=attention_scores,
        value_ptr=value_tensor,
        output_ptr=output,
        batch_size=batch_size,
        seq_len_q=seq_len_q,
        seq_len_k=seq_len_k,
        embed_dim=embed_dim,
        BLOCK_M=BLOCK_SIZE,
        BLOCK_N=1,
        BLOCK_K=64,
    )
    
    return output

def replacement_func():
    return optimized_fused_attention