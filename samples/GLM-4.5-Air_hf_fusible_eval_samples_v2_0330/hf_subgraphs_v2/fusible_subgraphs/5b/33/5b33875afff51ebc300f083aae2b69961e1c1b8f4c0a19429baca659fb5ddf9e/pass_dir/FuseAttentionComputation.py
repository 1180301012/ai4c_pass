import torch
import triton
import triton.language as tl

# Pattern matching function for attention computation: division + addition + softmax + dropout
def pattern(in_0, in_2):
    tmp_0 = in_0 / 8.0
    tmp_1 = tmp_0 + in_2
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.1, False, False)
    return tmp_3

# Argument extraction function
def replacement_args(in_0, in_2):
    return (in_0, in_2)

# Optimal Triton kernel for fused attention computation
@triton.jit
def fused_attention_kernel(
    attention_scores_ptr,
    attention_mask_ptr,
    attention_weights_ptr,
    H: tl.constexpr,
    T: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one head and one sequence position
    h_idx = tl.program_id(0)
    t_idx = tl.program_id(1)
    head_size = tl.program_id(2)
    
    # Calculate pointer offsets
    scores_base = h_idx * T + t_idx
    mask_base = h_idx * T + t_idx  # Same indexing for mask after broadcasting
    
    # Load attention scores for the current head and position
    # We load all T (sequence_length) elements for softmax
    offsets = tl.arange(0, T)
    mask_scores = offsets == tl.arange(0, T)
    
    scores = tl.load(attention_scores_ptr + scores_base * T + offsets, mask=mask_scores)
    masks = tl.load(attention_mask_ptr + mask_base * T + offsets, mask=mask_scores)
    
    # Fused computation: division + addition + softmax + dropout scaling
    # Since dropout is in inference mode, it's just scaling by 0.9
    scaled_scores = scores * 0.125  # / 8.0
    added_scores = scaled_scores + masks
    max_scores = tl.max(added_scores)
    shifted_scores = added_scores - max_scores
    exp_scores = tl.exp(shifted_scores)
    sum_exp_scores = tl.sum(exp_scores)
    softmax_scores = exp_scores / sum_exp_scores
    final_weights = softmax_scores * 0.9  # Dropout scaling (1-0.1)
    
    # Store result
    output_base = h_idx * T * T + t_idx * T
    tl.store(attention_weights_ptr + output_base + offsets, final_weights, mask=mask_scores)

@torch.fx.wrap
def fused_attention_computation(in_0, in_2):
    # Handle different batch sizes and head configurations
    B, nh, T, T_v = in_0.shape
    H = nh
    
    # Create output tensor
    attention_weights = torch.empty_like(in_0)
    
    # For simplicity, we'll process each head independently
    # In a more optimized version, we could batch this better
    blocks_per_head = (T + 127) // 128
    head_size = 1
    
    # Launch kernel for each head and each sequence position
    fused_attention_kernel[(H * T * head_size,)](
        attention_scores_ptr=in_0,
        attention_mask_ptr=in_2,
        attention_weights_ptr=attention_weights,
        H=H,
        T=T,
        BLOCK_SIZE=128,
    )
    
    return attention_weights

# Replacement function (returns function reference)
def replacement_func():
    return fused_attention_computation