import torch
import triton
import triton.language as tl
import math

def pattern(matmul, rel_logits_w, residual):
    # This pattern matches the complete attention computation chain
    tmp_1 = matmul.reshape(-1, 16, 31)
    tmp_2 = torch.nn.functional.pad(tmp_1, [0, 1], 'constant', None)
    tmp_3 = tmp_2.flatten(1)
    tmp_4 = torch.nn.functional.pad(tmp_3, [0, 15], 'constant', None)
    tmp_5 = tmp_4.reshape(-1, 17, 31)
    tmp_6 = tmp_5[(slice(None, None, None), slice(None, 16, None), slice(15, None, None))]
    tmp_7 = tmp_6.reshape(4, 16, 1, 16, 16)
    tmp_8 = tmp_7.expand(-1, -1, 16, -1, -1)
    tmp_9 = tmp_8.permute((0, 3, 1, 4, 2))
    tmp_10 = tmp_9 + rel_logits_w
    tmp_11 = tmp_10.reshape(4, 256, 256)
    tmp_12 = residual + tmp_11
    attention_scores = tmp_12
    return attention_scores

def replacement_args(matmul, rel_logits_w, residual):
    return (matmul, rel_logits_w, residual)

@triton.jit
def relative_position_bias_kernel(
    matmul_ptr,
    rel_logits_w_ptr,
    bias_ptr,
    batch_size,
    head_size,
    seq_len,
    window_size,
    BLOCK_K: tl.constexpr,
):
    """Compute relative position bias directly without complex reshape operations"""
    pid = tl.program_id(0)
    
    total_elements = batch_size * head_size * seq_len * seq_len
    element_idx = pid * 256 + tl.arange(0, 256)
    mask = element_idx < total_elements
    
    if not mask[0]:
        return
    
    # Map to output coordinates
    batch_idx = element_idx // (head_size * seq_len * seq_len) % batch_size
    head_idx = element_idx // (seq_len * seq_len) % head_size
    query_idx = element_idx // seq_len % seq_len
    key_idx = element_idx % seq_len
    
    # Compute relative position directly
    rel_pos = key_idx - query_idx
    
    # Clamp to window bounds
    rel_pos = tl.maximum(rel_pos, -window_size + 1)
    rel_pos = tl.minimum(rel_pos, window_size - 1)
    
    # Convert to bias tensor coordinates
    # The rel_logits_w has shape [batch_size, head_size, window_size, window_size]
    abs_query = tl.abs(query_idx)
    abs_key = tl.abs(key_idx)
    
    # Standard relative position bias computation
    bias_row = rel_pos + window_size - 1
    bias_col = window_size - 1
    
    # Load bias value - simplified version
    bias_offset = batch_idx * head_size * window_size * window_size + head_idx * window_size * window_size + bias_row * window_size + bias_col
    
    bias_val = tl.load(rel_logits_w_ptr + bias_offset, 
                      mask=(bias_row >= 0) & (bias_row < 2 * window_size - 1) & 
                           (bias_col >= 0) & (bias_col < 2 * window_size - 1))
    
    # Store the bias
    tl.store(bias_ptr + element_idx, bias_val, mask=mask)

@torch.fx.wrap
def optimized_attention_computation(matmul, rel_logits_w, residual):
    """Optimized attention computation with fused relative position bias"""
    batch_size, head_size, seq_len_q, seq_len_k = matmul.shape
    
    # For different head sizes (16x16 or 8x8), determine window size
    if seq_len_q == 16:
        window_size = 15
    elif seq_len_q == 8:
        window_size = 7
    else:
        raise ValueError(f"Unsupported sequence length: {seq_len_q}")
    
    # Create relative position bias tensor using optimized kernel
    bias_shape = (batch_size, head_size, seq_len_q, seq_len_k)
    bias = torch.zeros(bias_shape, dtype=matmul.dtype, device=matmul.device)
    
    # Launch relative position bias kernel
    total_elements = batch_size * head_size * seq_len_q * seq_len_k
    BLOCK_SIZE = 256
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    relative_position_bias_kernel[(num_programs,)](
        matmul_ptr=matmul,
        rel_logits_w_ptr=rel_logits_w,
        bias_ptr=bias,
        batch_size=batch_size,
        head_size=head_size,
        seq_len=seq_len_q,
        window_size=window_size,
        BLOCK_K=32,
    )
    
    # Add bias to residual
    attention_scores = residual + bias
    
    return attention_scores

def replacement_func():
    return optimized_attention_computation