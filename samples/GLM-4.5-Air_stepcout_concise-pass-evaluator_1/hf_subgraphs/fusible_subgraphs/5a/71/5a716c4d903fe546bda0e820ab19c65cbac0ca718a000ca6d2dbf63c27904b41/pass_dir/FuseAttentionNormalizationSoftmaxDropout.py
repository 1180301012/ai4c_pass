import torch
import triton
import triton.language as tl
import math

def pattern(in_0, in_1, in_2, in_3):
    # Match the exact computation pattern from the model
    tmp_0 = in_0 + in_3
    tmp_1 = tmp_0 + in_2
    tmp_2 = tmp_1 / 8.0
    tmp_3 = tmp_2 + in_1
    tmp_4 = torch.nn.functional.softmax(tmp_3, dim=-1)
    tmp_5 = torch.nn.functional.dropout(tmp_4, 0.1, False, False)
    # The pattern must include all intermediate values as the original computation
    return tmp_0, tmp_1, tmp_2, tmp_3, tmp_4, tmp_5

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_attention_kernel(
    in_0_ptr,
    in_1_ptr,
    in_2_ptr,
    in_3_ptr,
    out_ptr,
    num_heads,
    seq_len,
    batch_size,
    scale_factor: tl.constexpr,
    dropout_prob: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Grid setup: batch_size * num_heads blocks
    pid = tl.program_id(0)
    batch_id = pid // num_heads
    head_id = pid % num_heads
    
    # Each block handles a portion of the sequence
    m_offset = pid * BLOCK_M
    n_offset = tl.arange(0, BLOCK_N)
    
    # Create masks for bounds checking
    mask_n = n_offset < seq_len
    
    # Load all inputs with broadcasting support
    # in_0, in_2, in_3: [batch_size, num_heads, seq_len, seq_len]
    # in_1: [batch_size, 1, 1, seq_len] (broadcastable)
    
    # Base indices for batch and head
    in_0_base_idx = batch_id * num_heads * seq_len * seq_len + head_id * seq_len * seq_len + m_offset * seq_len
    in_2_base_idx = batch_id * num_heads * seq_len * seq_len + head_id * seq_len * seq_len + m_offset * seq_len
    in_3_base_idx = batch_id * num_heads * seq_len * seq_len + head_id * seq_len * seq_len + m_offset * seq_len
    
    # Load inputs
    in_0_vec = tl.load(in_0_ptr + in_0_base_idx + n_offset, mask_n, other=0.0)
    in_2_vec = tl.load(in_2_ptr + in_2_base_idx + n_offset, mask_n, other=0.0)
    in_3_vec = tl.load(in_3_ptr + in_3_base_idx + n_offset, mask_n, other=0.0)
    
    # Load in_1 with broadcasting - only load once per batch and position
    in_1_base_idx = batch_id * seq_len + n_offset
    in_1_vec = tl.load(in_1_ptr + in_1_base_idx, mask_n, other=0.0)
    
    # Compute fused operations: (in_0 + in_3 + in_2 + in_1) / 8.0
    sum_val = (in_0_vec + in_2_vec + in_3_vec + in_1_vec) / scale_factor
    
    # Apply softmax
    max_val = tl.max(sum_val, keepdims=True)
    exp_val = tl.exp(sum_val - max_val)
    sum_exp = tl.sum(exp_val, keepdims=True)
    softmax_output = exp_val / sum_exp
    
    # Apply dropout
    dropout_mask = tl.rand(tl.shape_to(softmax_output)) > dropout_prob
    dropout_output = softmax_output * dropout_mask / (1.0 - dropout_prob)
    
    # Store result
    out_base_idx = batch_id * num_heads * seq_len * seq_len + head_id * seq_len * seq_len + m_offset * seq_len
    tl.store(out_ptr + out_base_idx + n_offset, dropout_output, mask=mask_n)

@torch.fx.wrap
def fused_attention_forward(in_0, in_1, in_2, in_3):
    batch_size, num_heads, seq_len, _ = in_0.shape
    
    # Determine optimal block sizes based on sequence length
    if seq_len <= 128:
        BLOCK_M = 32
        BLOCK_N = 64
    elif seq_len <= 512:
        BLOCK_M = 64
        BLOCK_N = 64
    else:
        BLOCK_M = 128
        BLOCK_N = 128
    
    # Calculate grid size
    total_blocks = batch_size * num_heads
    grid = (total_blocks,)
    
    # Create intermediate and output tensors
    tmp_0 = torch.empty_like(in_0)  # in_0 + in_3
    tmp_1 = torch.empty_like(in_0)  # tmp_0 + in_2  
    tmp_2 = torch.empty_like(in_0)  # tmp_1 / 8.0
    tmp_3 = torch.empty_like(in_0)  # tmp_2 + in_1
    tmp_4 = torch.empty_like(in_0)  # softmax(tmp_3, dim=-1)
    tmp_5 = torch.empty_like(in_0)  # dropout(tmp_4, 0.1, False, False)
    
    # Launch kernel with fused operations
    fused_attention_kernel[grid](
        in_0_ptr=in_0,
        in_1_ptr=in_1.flatten(),  # Flatten for easier indexing
        in_2_ptr=in_2,
        in_3_ptr=in_3,
        out_ptr=tmp_5,
        num_heads=num_heads,
        seq_len=seq_len,
        batch_size=batch_size,
        scale_factor=8.0,
        dropout_prob=0.1,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    
    # Compute intermediate values separately for now
    # TODO: In a full fusion, compute all in one kernel
    tmp_0 = in_0 + in_3
    tmp_1 = tmp_0 + in_2
    tmp_2 = tmp_1 / 8.0
    tmp_3 = tmp_2 + in_1
    tmp_4 = torch.nn.functional.softmax(tmp_3, dim=-1)
    
    # Return all intermediate values as expected by pattern
    return tmp_0, tmp_1, tmp_2, tmp_3, tmp_4, tmp_5

def replacement_func():
    return fused_attention_forward