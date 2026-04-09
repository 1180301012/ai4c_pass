import torch
import triton
import triton.language as tl

def pattern(x, y, z):
    tmp_0 = torch.matmul(x, y)
    tmp_1 = tmp_0 / 5.656854249492381
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.0, False, False)
    tmp_4 = torch.matmul(tmp_3, z)
    tmp_5 = tmp_4.permute(0, 2, 1, 3)
    tmp_6 = tmp_5.contiguous()
    # Use flexible view pattern - should match different output shapes
    output_shape = (1, -1, tmp_6.shape[-1])  # Flexible shape
    tmp_7 = tmp_6.view(output_shape)
    return (tmp_7,)

def replacement_args(x, y, z):
    return (x, y, z)



@torch.fx.wrap
def optimized_fused_attention(query, key, value):
    # Scale factor for this specific pattern
    scale_factor = 1.0 / 5.656854249492381
    
    # Get dimensions and prepare intermediate tensors using only allowed APIs
    batch_size, seq_len, head_dim_1, head_dim_2 = query.shape
    query_reshaped = query.reshape(batch_size * seq_len, head_dim_1)
    key_reshaped = key.reshape(batch_size * seq_len, head_dim_2)
    value_reshaped = value.reshape(batch_size * seq_len, head_dim_2)
    
    # Prepare output tensor
    output_size = batch_size * seq_len * head_dim_2
    output = torch.empty(output_size, dtype=query.dtype, device=query.device)
    
    # Launch Triton kernel with simpler approach
    total_queries = batch_size * seq_len
    BLOCK_SIZE = 256
    
    grid = (total_queries + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    simplified_attention_kernel[grid](
        query_reshaped, key_reshaped, value_reshaped,
        output,
        total_queries, head_dim_1, head_dim_2,
        scale_factor,
        BLOCK_SIZE
    )
    
    # Reshape back to expected format
    final_output = output.reshape(batch_size, seq_len, head_dim_2)
    
    return (final_output,)

@triton.jit
def simplified_attention_kernel(
    query_ptr, key_ptr, value_ptr, output_ptr,
    total_queries, k_dim, v_dim,
    scale_factor,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Process queries in blocks
    query_start = pid * BLOCK_SIZE
    query_end = min(query_start + BLOCK_SIZE, total_queries)
    
    for idx in range(query_start, query_end):
        # Load single query
        query = tl.load(query_ptr + idx * k_dim)
        
        # Attention computation for this query
        attention_scores = tl.zeros(v_dim, dtype=tl.float32)
        
        for k in range(k_dim):
            key_val = tl.load(key_ptr + idx * k_dim + k)
            for v in range(v_dim):
                value_val = tl.load(value_ptr + idx * k_dim + v)
                attention_scores[v] += query[k] * key_val * scale_factor
        
        # Softmax
        max_score = tl.max(attention_scores)
        exp_scores = tl.exp(attention_scores - max_score)
        sum_exp = tl.sum(exp_scores)
        attention_weights = exp_scores / sum_exp
        
        # Weighted sum of values
        output = tl.zeros(v_dim, dtype=tl.float32)
        for v in range(v_dim):
            output += attention_weights[v] * tl.load(value_ptr + idx * k_dim + v)
        
        # Store result
        for v in range(v_dim):
            tl.store(output_ptr + idx * v_dim + v, output[v])

def replacement_func():
    return optimized_fused_attention