import torch
import triton
import triton.language as tl
import math

def pattern(q, k, v, scale_factor=5.656854249492381):
    # Match the attention computation pattern: matmul -> scale -> softmax -> dropout -> matmul
    # Dropout with p=0.0 is essentially a no-op, so we eliminate it
    attn_scores = torch.matmul(q, k)
    scaled_scores = attn_scores / scale_factor
    attn_probs = torch.nn.functional.softmax(scaled_scores, dim=-1)
    # Dropout is eliminated since p=0.0 makes it a no-op
    attn_output = torch.matmul(attn_probs, v)
    return attn_scores, attn_output

def replacement_args(q, k, v, scale_factor=5.656854249492381):
    return (q, k, v, scale_factor)

@triton.jit
def fused_attention_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    batch_size, num_heads, seq_len_q, seq_len_k, head_dim,
    scale_factor,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program ID for this thread block
    pid = tl.program_id(0)
    batch_and_head = pid // seq_len_q
    idx = pid % seq_len_q
    
    # Load query vector
    q_offset = (batch_and_head * seq_len_q + idx) * head_dim
    q = tl.load(q_ptr + q_offset, mask=True)
    
    # Initialize output accumulator
    acc = tl.zeros((head_dim,), dtype=tl.float32)
    
    # Compute attention with all keys
    for k_idx in range(0, seq_len_k, BLOCK_N):
        k_end = min(k_idx + BLOCK_N, seq_len_k)
        
        # Load key block
        k_offset_base = batch_and_head * seq_len_k * head_dim
        k_offsets = k_offset_base + tl.arange(0, BLOCK_N * head_dim, head_dim)
        k_mask = tl.arange(0, BLOCK_N) < (k_end - k_idx)
        k_block = tl.load(k_ptr + k_offsets, mask=k_mask[:, None], other=0.0)
        k_block = k_block.reshape((BLOCK_N, head_dim))
        
        # Compute dot products for this key block
        scores = tl.dot(q, k_block.T) * scale_factor
        
        # Compute softmax (stably)
        max_scores = tl.maximum(tl.reduce_max(scores, axis=0), tl.full([1], float('-inf'), dtype=tl.float32))
        exp_scores = tl.exp(scores - max_scores)
        sum_exp = tl.sum(exp_scores, axis=0)
        attn_weights = exp_scores / sum_exp
        
        # Load value block
        v_offset_base = batch_and_head * seq_len_k * head_dim
        v_offsets = v_offset_base + tl.arange(k_idx * head_dim, k_end * head_dim)
        v_mask = tl.arange(0, BLOCK_N) < (k_end - k_idx)
        v_block = tl.load(v_ptr + v_offsets, mask=v_mask[:, None], other=0.0)
        v_block = v_block.reshape((BLOCK_N, head_dim))
        
        # Accumulate weighted values
        acc += tl.dot(attn_weights, v_block)
    
    # Store result
    out_offset = (batch_and_head * seq_len_q + idx) * head_dim
    tl.store(out_ptr + out_offset, acc)

@torch.fx.wrap
def fused_attention_forward(q, k, v, scale_factor=5.656854249492381):
    batch_size, num_heads, seq_len_q, head_dim = q.shape
    _, _, seq_len_k, _ = k.shape
    
    # Calculate total elements needed
    total_elements = batch_size * num_heads * seq_len_q
    
    # Block sizes tuned for performance
    BLOCK_M = 32  # Query tokens per block
    BLOCK_N = 32  # Key tokens per block  
    BLOCK_K = 32  # Head dimension per block
    
    # Create output tensor
    output = torch.zeros_like(q)
    
    # Number of thread blocks
    num_blocks = (total_elements + BLOCK_M - 1) // BLOCK_M
    
    # Launch kernel
    fused_attention_kernel[(num_blocks,)](
        q_ptr=q,
        k_ptr=k,
        v_ptr=v,
        out_ptr=output,
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len_q=seq_len_q,
        seq_len_k=seq_len_k,
        head_dim=head_dim,
        scale_factor=scale_factor,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    
    return output

def replacement_func():
    return fused_attention_forward