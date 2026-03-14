import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    """
    Pattern to match attention score computation:
    - in_0: attention_scores [B, H, S, S]
    - in_1: extended_attention_mask [B, 1, 1, S]
    - in_2: relative_position_scores_key [B, H, S, S]
    - in_3: relative_position_scores_query [B, H, S, S]
    """
    tmp_0 = in_0 + in_3
    tmp_1 = tmp_0 + in_2
    tmp_2 = tmp_1 / 8.0
    tmp_3 = tmp_2 + in_1
    tmp_4 = torch.nn.functional.softmax(tmp_3, dim=-1)
    tmp_5 = torch.nn.functional.dropout(tmp_4, 0.1, False, False)
    return (tmp_5,)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['seq_len'],
)
@triton.jit
def fused_attention_softmax_kernel(
    in_0_ptr,        # attention_scores [B, H, S, S]
    in_1_ptr,        # extended_attention_mask [B, 1, 1, S]
    in_2_ptr,        # relative_position_scores_key [B, H, S, S]
    in_3_ptr,        # relative_position_scores_query [B, H, S, S]
    out_ptr,         # output [B, H, S, S]
    batch_size,      # B
    num_heads,       # H
    seq_len,         # S
    mask_stride_b,   # stride for mask batch dimension
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row of the attention matrix
    # row_idx = batch * H * S + head * S + row
    row_idx = tl.program_id(0)
    
    # Calculate batch, head, and row indices
    batch_head_size = num_heads * seq_len
    batch_idx = row_idx // batch_head_size
    remainder = row_idx % batch_head_size
    head_idx = remainder // seq_len
    row_in_seq = remainder % seq_len
    
    # Base offset for this row in the 4D tensors [B, H, S, S]
    row_offset = batch_idx * num_heads * seq_len * seq_len + head_idx * seq_len * seq_len + row_in_seq * seq_len
    
    # Mask offset: [B, 1, 1, S] - only varies by batch and column
    mask_offset = batch_idx * mask_stride_b
    
    # Process the row in blocks
    # First pass: compute max for numerical stability
    row_max = -float('inf')
    for block_start in range(0, seq_len, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < seq_len
        
        # Load values
        in_0_vals = tl.load(in_0_ptr + row_offset + col_offsets, mask=mask, other=0.0)
        in_2_vals = tl.load(in_2_ptr + row_offset + col_offsets, mask=mask, other=0.0)
        in_3_vals = tl.load(in_3_ptr + row_offset + col_offsets, mask=mask, other=0.0)
        in_1_vals = tl.load(in_1_ptr + mask_offset + col_offsets, mask=mask, other=0.0)
        
        # Compute: (in_0 + in_3 + in_2) / 8.0 + in_1
        combined = (in_0_vals + in_3_vals + in_2_vals) / 8.0 + in_1_vals
        
        # Update max (use a large negative value for masked positions)
        combined = tl.where(mask, combined, -float('inf'))
        block_max = tl.max(combined, axis=0)
        row_max = tl.maximum(row_max, block_max)
    
    # Second pass: compute sum of exp(x - max)
    row_sum = 0.0
    for block_start in range(0, seq_len, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < seq_len
        
        # Load and compute again
        in_0_vals = tl.load(in_0_ptr + row_offset + col_offsets, mask=mask, other=0.0)
        in_2_vals = tl.load(in_2_ptr + row_offset + col_offsets, mask=mask, other=0.0)
        in_3_vals = tl.load(in_3_ptr + row_offset + col_offsets, mask=mask, other=0.0)
        in_1_vals = tl.load(in_1_ptr + mask_offset + col_offsets, mask=mask, other=0.0)
        
        combined = (in_0_vals + in_3_vals + in_2_vals) / 8.0 + in_1_vals
        
        # Compute exp(x - max)
        exp_vals = tl.exp(combined - row_max)
        exp_vals = tl.where(mask, exp_vals, 0.0)
        row_sum += tl.sum(exp_vals, axis=0)
    
    # Third pass: compute softmax and store
    for block_start in range(0, seq_len, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < seq_len
        
        # Load and compute again
        in_0_vals = tl.load(in_0_ptr + row_offset + col_offsets, mask=mask, other=0.0)
        in_2_vals = tl.load(in_2_ptr + row_offset + col_offsets, mask=mask, other=0.0)
        in_3_vals = tl.load(in_3_ptr + row_offset + col_offsets, mask=mask, other=0.0)
        in_1_vals = tl.load(in_1_ptr + mask_offset + col_offsets, mask=mask, other=0.0)
        
        combined = (in_0_vals + in_3_vals + in_2_vals) / 8.0 + in_1_vals
        
        # Compute softmax: exp(x - max) / sum
        exp_vals = tl.exp(combined - row_max)
        softmax_vals = exp_vals / row_sum
        
        # Store result
        tl.store(out_ptr + row_offset + col_offsets, softmax_vals, mask=mask)


@torch.fx.wrap
def fused_attention_softmax(in_0, in_1, in_2, in_3):
    """
    Fused kernel for attention score computation:
    softmax((in_0 + in_3 + in_2) / 8.0 + in_1, dim=-1)
    Dropout with training=False is a no-op
    """
    # Get dimensions
    B, H, S, _ = in_0.shape
    
    # Ensure contiguous
    in_0 = in_0.contiguous()
    in_1 = in_1.contiguous()
    in_2 = in_2.contiguous()
    in_3 = in_3.contiguous()
    
    # Output tensor
    out = torch.empty_like(in_0)
    
    # Total number of rows to process
    num_rows = B * H * S
    
    # Mask stride for batch dimension
    mask_stride_b = in_1.stride(0)
    
    # Launch kernel - one program per row
    grid = (num_rows,)
    
    fused_attention_softmax_kernel[grid](
        in_0,
        in_1,
        in_2,
        in_3,
        out,
        B,
        H,
        S,
        mask_stride_b,
    )
    
    return (out,)


def replacement_func():
    return fused_attention_softmax