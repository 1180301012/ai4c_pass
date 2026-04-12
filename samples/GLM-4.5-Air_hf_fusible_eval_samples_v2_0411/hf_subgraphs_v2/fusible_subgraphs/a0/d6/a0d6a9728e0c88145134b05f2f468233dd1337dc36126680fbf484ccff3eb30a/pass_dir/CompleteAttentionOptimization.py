import torch
import triton
import triton.language as tl

# This pass matches the complete computation pattern for attention + reshape
def pattern(query, key, value):
    # Complete pattern matching the target computation
    bmm = torch.bmm(query, key)  # query @ key computation
    
    # Softmax + dropout (can be optimized in the kernel)
    tmp_1 = torch.nn.functional.softmax(bmm, dim=-1)
    tmp_2 = torch.nn.functional.dropout(tmp_1, p=0.0, training=False)
    
    # Second BMM for attention output
    bmm_1 = torch.bmm(tmp_2, value)
    
    # Reshape sequence that can be fused
    tmp_4 = bmm_1.view(1, bmm_1.shape[0], 1, bmm_1.shape[2])
    tmp_5 = tmp_4.transpose(1, 2)
    tmp_6 = tmp_5.reshape(1, 1, bmm_1.shape[0] * bmm_1.shape[2])
    
    # Return only the final result (the observable output)
    return tmp_6

def replacement_args(query, key, value):
    return (query, key, value)

# Optimized Triton kernel for complete attention computation
@triton.jit
def complete_attention_kernel(
    query_ptr, key_ptr, value_ptr, output_ptr,
    batch_size, num_heads, head_dim,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(0)
    batch_idx = pid // num_heads
    head_idx = pid % num_heads
    
    # Calculate pointers for this head
    query_offset = batch_idx * num_heads * head_dim + head_idx * head_dim
    key_offset = batch_idx * num_heads * head_dim + head_idx * head_dim
    value_offset = batch_idx * num_heads * head_dim + head_idx * head_dim
    
    query_ptr = query_ptr + query_offset
    key_ptr = key_ptr + key_offset
    value_ptr = value_ptr + value_offset
    output_ptr = output_ptr + batch_idx * num_heads * head_dim + head_idx * head_dim
    
    # Accumulators (first for attention scores, then for final output)
    attn_scores = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    final_output = tl.zeros((BLOCK_SIZE_K,), dtype=tl.float32)
    
    # Compute attention scores: query @ key.transpose(-2, -1)
    for k in range(0, head_dim, BLOCK_SIZE_K):
        query_vec = tl.load(query_ptr + k * head_dim + tl.arange(0, BLOCK_SIZE_K), 
                           mask=k + tl.arange(0, BLOCK_SIZE_K) < head_dim, other=0.0)
        
        for n in range(0, 1):  # key has shape [B, H, 1] in our case
            key_val = tl.load(key_ptr + n * head_dim + k, mask=k < head_dim, other=0.0)
            attn_score = 0.0
            ki = tl.arange(0, BLOCK_SIZE_K)
            if k + ki < head_dim:
                attn_score = query_vec * key_val
            store_idx = n * BLOCK_SIZE_N + ki
            if store_idx < BLOCK_SIZE_N:
                tl.store(attn_scores + store_idx, attn_score, mask=ki < BLOCK_SIZE_N)
    
    # Simplified softmax for small dimension
    max_score = tl.max(attn_scores)
    exp_score = tl.exp(attn_scores - max_score)  # Numerically stable softmax
    sum_exp = tl.sum(exp_score)
    attn_weights = exp_score / sum_exp
    
    # Compute final output: attn_weights @ value
    for k in range(0, head_dim, BLOCK_SIZE_K):
        for n in range(0, 1):  # attn_weights has shape [B, H, 1]
            weight = tl.load(attn_weights + n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N),
                            mask=tl.arange(0, BLOCK_SIZE_N) < BLOCK_SIZE_N, other=0.0)[0]
            val_vec = tl.load(value_ptr + n * head_dim + k * head_dim + tl.arange(0, BLOCK_SIZE_K),
                            mask=n * head_dim + k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K) < batch_size * num_heads * head_dim, other=0.0)
            ki = tl.arange(0, BLOCK_SIZE_K)
            if k + ki < head_dim:
                final_output += weight * val_vec
    
    # Store final output (will be reshaped later)
    tl.store(output_ptr + tl.arange(0, BLOCK_SIZE_K), final_output, mask=tl.arange(0, BLOCK_SIZE_K) < head_dim)

@triton.jit
def reshape_kernel(
    input_ptr, output_ptr,
    total_elements, final_size,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < final_size
    
    # Direct copy from flattened input to final output
    tl.store(output_ptr + offsets, tl.load(input_ptr + offsets, mask=offsets < total_elements, other=0.0), mask=mask)

# Complete attention optimization function at module level
@torch.fx.wrap
def complete_attention_optimization(query, key, value):
    batch_size, num_heads, head_dim = query.shape[0], query.shape[1], query.shape[2]
    
    # Allocate intermediate output
    intermediate_output = torch.empty((batch_size, num_heads, head_dim), dtype=query.dtype, device=query.device)
    
    # Launch optimized attention kernel
    total_heads = batch_size * num_heads
    BLOCK_SIZE_M = 1  # Process one head at a time
    BLOCK_SIZE_N = 1  # Attention output size (typically 1x1 for our case)
    BLOCK_SIZE_K = min(32, head_dim)  # Optimized tile size
    
    complete_attention_kernel[(
        (total_heads + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
    )](query, key, value, intermediate_output,
        batch_size, num_heads, head_dim,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)
    
    # Direct reshape to final format [1, 1, batch_size * head_dim] with Triton kernel
    final_size = batch_size * head_dim
    output = torch.empty(1, 1, final_size, dtype=query.dtype, device=query.device)
    
    BLOCK_SIZE = 1024
    num_blocks = (final_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Flatten intermediate output and reshape directly
    flattened = intermediate_output.flatten()
    reshape_kernel[(num_blocks,)](
        flattened, output,
        flattened.numel(), final_size,
        BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return complete_attention_optimization