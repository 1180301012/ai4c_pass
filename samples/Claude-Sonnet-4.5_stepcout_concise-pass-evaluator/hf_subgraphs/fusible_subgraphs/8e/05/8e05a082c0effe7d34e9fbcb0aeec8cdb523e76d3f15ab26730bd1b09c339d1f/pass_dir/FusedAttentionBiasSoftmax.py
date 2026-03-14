import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    """
    Match the pattern:
    1. Index lookup from bias table
    2. Reshape and permute
    3. Add to attention scores
    4. Add attention mask
    5. Softmax
    6. Dropout (p=0.0, which is identity)
    """
    tmp_0 = in_0
    tmp_1 = tmp_0[in_3]
    tmp_2 = tmp_1.view(144, 144, -1)
    tmp_3 = tmp_2.permute(2, 0, 1)
    tmp_4 = tmp_3.contiguous()
    tmp_5 = tmp_4.unsqueeze(0)
    tmp_6 = in_1 + tmp_5
    tmp_7 = tmp_6.view(1, 64, 4, 144, 144)
    tmp_8 = in_2.unsqueeze(1)
    tmp_9 = tmp_8.unsqueeze(0)
    tmp_10 = tmp_7 + tmp_9
    tmp_11 = tmp_10.view(-1, 4, 144, 144)
    tmp_12 = torch.nn.functional.softmax(tmp_11, dim=-1)
    tmp_13 = torch.nn.functional.dropout(tmp_12, 0.0, False, False)
    return tmp_13

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_attention_bias_softmax_kernel(
    bias_table_ptr,
    indices_ptr,
    attn_scores_ptr,
    attn_mask_ptr,
    output_ptr,
    num_indices,
    num_heads,
    seq_len,
    batch_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for:
    - Index lookup from bias table
    - Add bias to attention scores
    - Add attention mask
    - Softmax along last dimension
    """
    # Get program ID
    pid = tl.program_id(0)
    
    # Each program handles one row of the final [batch*heads, seq_len, seq_len] tensor
    batch_idx = pid // (num_heads * seq_len)
    head_idx = (pid // seq_len) % num_heads
    row_idx = pid % seq_len
    
    # Calculate offsets for this row
    row_start = pid * seq_len
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < seq_len
    
    # Load attention scores for this row
    attn_ptr = attn_scores_ptr + batch_idx * num_heads * seq_len * seq_len + head_idx * seq_len * seq_len + row_idx * seq_len
    attn = tl.load(attn_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=-float('inf'))
    
    # Load bias values for this row
    # Bias shape after indexing and permute: [num_heads, seq_len, seq_len]
    bias_row_offset = head_idx * seq_len * seq_len + row_idx * seq_len
    bias_indices_offset = bias_row_offset + tl.arange(0, BLOCK_SIZE)
    
    # For each position in the row, we need to index into the bias table
    # This is complex due to the reshape/permute, so we'll do a simpler approach
    # and load the already-gathered bias
    
    # Actually, let's simplify: we'll compute the bias index for each position
    bias_idx = row_idx * seq_len + tl.arange(0, BLOCK_SIZE)
    bias_idx = tl.where(mask, bias_idx, 0)
    
    # Load from indices array
    actual_indices = tl.load(indices_ptr + bias_idx, mask=mask, other=0)
    
    # Load bias values using the indices
    # Bias table shape: [num_bias_entries, num_heads]
    bias_vals = tl.load(bias_table_ptr + actual_indices * num_heads + head_idx, mask=mask, other=0.0)
    
    # Add bias to attention scores
    attn = attn + bias_vals
    
    # Load and add attention mask
    mask_ptr = attn_mask_ptr + batch_idx * seq_len * seq_len + row_idx * seq_len
    attn_mask_vals = tl.load(mask_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    attn = attn + attn_mask_vals
    
    # Softmax: subtract max for numerical stability
    row_max = tl.max(attn, axis=0)
    attn = attn - row_max
    
    # Exponentiate
    exp_attn = tl.exp(attn)
    
    # Sum and normalize
    exp_sum = tl.sum(exp_attn, axis=0)
    output = exp_attn / exp_sum
    
    # Store result
    output_ptr_offset = row_start + tl.arange(0, BLOCK_SIZE)
    tl.store(output_ptr + output_ptr_offset, output, mask=mask)

@torch.fx.wrap
def fused_attention_bias_softmax(bias_table, attn_scores, attn_mask, indices):
    """
    Fused implementation that combines:
    1. Index lookup and reshape
    2. Bias addition
    3. Mask addition
    4. Softmax
    """
    # Get dimensions from inputs
    # attn_scores shape: [batch, heads, seq_len, seq_len]
    batch, heads, seq_len, _ = attn_scores.shape
    
    # Flatten batch and heads for processing
    total_rows = batch * heads * seq_len
    
    # Allocate output
    output = torch.empty_like(attn_scores)
    
    # Choose block size based on seq_len
    if seq_len <= 64:
        BLOCK_SIZE = 64
    elif seq_len <= 128:
        BLOCK_SIZE = 128
    elif seq_len <= 256:
        BLOCK_SIZE = 256
    else:
        BLOCK_SIZE = 512
    
    # Launch kernel
    grid = (total_rows,)
    
    fused_attention_bias_softmax_kernel[grid](
        bias_table_ptr=bias_table,
        indices_ptr=indices,
        attn_scores_ptr=attn_scores,
        attn_mask_ptr=attn_mask,
        output_ptr=output,
        num_indices=indices.numel(),
        num_heads=heads,
        seq_len=seq_len,
        batch_size=batch,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_attention_bias_softmax