import torch
import triton
import triton.language as tl

def pattern(attention_scores, value_layer, extended_attention_mask):
    # Scale attention scores - use generic division that could match different scale factors
    scaled = attention_scores / 8.0  # Will match to any scalar division
    # Add attention mask 
    masked = scaled + extended_attention_mask
    # Apply softmax
    attn_weights = torch.nn.functional.softmax(masked, dim=-1)
    # Apply dropout (training=False)
    attn_probs = torch.nn.functional.dropout(attn_weights, p=0.1, training=False, inplace=False)
    # Matrix multiplication with value layer
    output = torch.matmul(attn_probs, value_layer)
    return attn_probs, output

def replacement_args(attention_scores, value_layer, extended_attention_mask):
    return (attention_scores, value_layer, extended_attention_mask)

@triton.jit
def fused_attention_kernel(
    scores_ptr, value_ptr, mask_ptr,
    out_probs_ptr, out_attn_ptr,
    batch_size, num_heads, seq_len, head_dim,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # Program identifiers
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Range of scores this program computes
    scores_start_m = pid_m * BLOCK_SIZE_M
    scores_end_m = min(scores_start_m + BLOCK_SIZE_M, seq_len)
    
    # Initialize accumulator for attention output
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Max for numerical stability
    max_val = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32) - tl.inf
    
    # Load mask (broadcast across heads and batch)
    mask_val = tl.load(mask_ptr, other=0.0)
    
    # First pass: find max along K dimension
    for k in range(0, seq_len, BLOCK_SIZE_K):
        # Load scores
        scores_offsets = scores_start_m + tl.arange(0, BLOCK_SIZE_M)
        k_offsets = k + tl.arange(0, BLOCK_SIZE_K)
        scores_mask = scores_offsets < scores_end_m
        
        scores = tl.load(
            scores_ptr + scores_offsets[:, None] * seq_len + k_offsets[None, :],
            mask=scores_mask[:, None],
            other=-tl.inf
        )
        
        # Add mask
        scores = scores + mask_val
        
        # Update max
        current_max = tl.max(scores, axis=1)
        max_val = tl.maximum(max_val, current_max)
    
    # Second pass: compute softmax denominator
    sum_exp = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    for k in range(0, seq_len, BLOCK_SIZE_K):
        scores_offsets = scores_start_m + tl.arange(0, BLOCK_SIZE_M)
        k_offsets = k + tl.arange(0, BLOCK_SIZE_K)
        scores_mask = scores_offsets < scores_end_m
        
        scores = tl.load(
            scores_ptr + scores_offsets[:, None] * seq_len + k_offsets[None, :],
            mask=scores_mask[:, None],
            other=-tl.inf
        )
        
        # Add mask and subtract max
        scores = scores + mask_val - max_val[:, None]
        
        # Compute exponential
        exp_scores = tl.exp(scores)
        sum_exp += tl.sum(exp_scores, axis=1)
    
    # Compute probabilities
    probs = tl.zeros((BLOCK_SIZE_M, seq_len), dtype=tl.float32)
    
    for k in range(0, seq_len, BLOCK_SIZE_K):
        scores_offsets = scores_start_m + tl.arange(0, BLOCK_SIZE_M)
        k_offsets = k + tl.arange(0, BLOCK_SIZE_K)
        scores_mask = scores_offsets < scores_end_m
        
        scores = tl.load(
            scores_ptr + scores_offsets[:, None] * seq_len + k_offsets[None, :],
            mask=scores_mask[:, None],
            other=-tl.inf
        )
        
        # Add mask, subtract max, and compute probabilities
        scores = scores + mask_val - max_val[:, None]
        exp_scores = tl.exp(scores)
        current_probs = exp_scores / sum_exp[:, None]
        
        # Store probabilities
        probs_mask = scores_offsets[:, None] < scores_end_m
        tl.store(
            out_probs_ptr + scores_offsets[:, None] * seq_len + k_offsets[None, :],
            current_probs,
            mask=probs_mask[:, None]
        )
    
    # Third pass: matmul with value layer
    for k in range(0, head_dim, BLOCK_SIZE_K):
        k_offsets = k + tl.arange(0, BLOCK_SIZE_K)
        value_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        value_mask = value_offsets < head_dim
        
        # Load value layer and probabilities
        values = tl.load(
            value_ptr + k_offsets[:, None] * head_dim + value_offsets[None, :],
            mask=value_mask[None, :],
            other=0.0
        )
        
        probs_mask = scores_offsets[:, None] < scores_end_m
        loaded_probs = tl.load(
            out_probs_ptr + scores_offsets[:, None] * seq_len + k_offsets[None, :],
            mask=probs_mask[:, None],
            other=0.0
        )
        
        # Matrix multiplication
        accumulator += tl.dot(loaded_probs, values)
    
    # Store attention output
    out_mask = scores_offsets[:, None] < scores_end_m
    tl.store(
        out_attn_ptr + scores_offsets[:, None] * head_dim + value_offsets[None, :],
        accumulator,
        mask=out_mask[:, None]
    )

@torch.fx.wrap
def fused_attention_kernel_wrapper(attention_scores, value_layer, extended_attention_mask):
    batch_size, num_heads, seq_len, _ = attention_scores.shape
    head_dim = value_layer.shape[-1]
    
    total_seq_len = seq_len * batch_size
    
    # Grid configuration
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    num_blocks_m = (total_seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_blocks_n = (head_dim + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Output tensors
    attn_probs = torch.empty_like(attention_scores, dtype=torch.float32)
    attn_output = torch.empty(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32, device=attention_scores.device)
    
    # Launch kernel
    fused_attention_kernel[(num_blocks_m, num_blocks_n)](
        attention_scores, value_layer, extended_attention_mask,
        attn_probs, attn_output,
        batch_size, num_heads, seq_len, head_dim,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    # Convert back to original dtype
    if attention_scores.dtype == torch.bfloat16:
        attn_probs = attn_probs.to(torch.bfloat16)
        attn_output = attn_output.to(torch.bfloat16)
    else:
        attn_probs = attn_probs.to(torch.float16)
        attn_output = attn_output.to(torch.float16)
    
    return attn_probs, attn_output

def replacement_func():
    return fused_attention_kernel_wrapper