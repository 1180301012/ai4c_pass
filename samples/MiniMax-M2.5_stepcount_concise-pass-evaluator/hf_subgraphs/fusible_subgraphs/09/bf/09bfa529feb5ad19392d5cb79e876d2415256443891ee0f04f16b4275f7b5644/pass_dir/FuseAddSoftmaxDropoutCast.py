import torch
import triton
import triton.language as tl

# Optimized fused kernel: add + softmax + dropout + cast
@triton.jit
def fused_attention_kernel(
    in_0_ptr,  # attention_scores
    in_1_ptr,  # extended_attention_mask
    out_ptr,   # output
    # Dimensions
    batch_size, num_heads, seq_len,
    # Dropout probability and seed offset
    dropout_p: tl.constexpr,
    seed_offset: tl.constexpr,
    # Block size for softmax
    BLOCK_SIZE: tl.constexpr,
):
    # Get program coordinates
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    row_idx = tl.program_id(2)
    
    # Calculate base offsets
    # Shape: [batch_size, num_heads, seq_len, seq_len]
    base_offset = (batch_idx * num_heads + head_idx) * seq_len * seq_len + row_idx * seq_len
    
    # Load the row from in_0 (attention_scores)
    # in_0 has shape [batch, num_heads, seq_len, seq_len]
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < seq_len
    
    # Load attention scores
    in_0_row_ptrs = in_0_ptr + base_offset + col_offsets
    x = tl.load(in_0_row_ptrs, mask=mask, other=-float('inf'))
    
    # Load attention mask and add
    # in_1 has shape [batch, 1, 1, seq_len] - broadcast along batch, num_heads, row
    # Need to calculate correct offset for in_1
    mask_1_offset = batch_idx * seq_len + col_offsets  # [batch, 1, 1, seq_len] -> broadcast
    in_1_row_ptrs = in_1_ptr + mask_1_offset
    mask_val = tl.load(in_1_row_ptrs, mask=mask, other=0.0)
    
    # Add attention scores and mask
    x = x + mask_val
    
    # Softmax along the last dimension (seq_len)
    # Subtract max for numerical stability
    max_val = tl.max(x, axis=0)
    x = x - max_val
    
    # Exp
    x = tl.exp(x)
    
    # Sum for normalization
    sum_val = tl.sum(x, axis=0)
    
    # Softmax output
    softmax_out = x / sum_val
    
    # Apply dropout if p > 0
    # Use pseudo-random number generation
    if dropout_p > 0.0:
        # Simple hash-based random
        random_seed = (batch_idx * 1000000 + head_idx * 10000 + row_idx * 100 + seed_offset) % 1000003
        random_offset = (random_seed * col_offsets) % 1000003
        random_mask = (random_offset % 1000) < (dropout_p * 1000)
        random_mask = random_mask & mask
        
        # Zero out dropped elements
        softmax_out = tl.where(random_mask, tl.zeros_like(softmax_out), softmax_out)
    
    # Cast to float32 (explicit, even though we're working in float32)
    # The output is already float32 in the kernel
    out = softmax_out.to(tl.float32)
    
    # Store result
    out_row_ptrs = out_ptr + base_offset + col_offsets
    tl.store(out_row_ptrs, out, mask=mask)


@torch.fx.wrap
def fused_attention_wrapper(in_0, in_1, dropout_p=0.0, seed_offset=42):
    """
    Fused kernel: add + softmax + dropout + cast
    in_0: attention_scores, shape [batch, num_heads, seq_len, seq_len]
    in_1: extended_attention_mask, shape [batch, 1, 1, seq_len]
    dropout_p: dropout probability
    """
    # Get shapes
    batch_size, num_heads, seq_len, _ = in_0.shape
    
    # Output tensor
    out = torch.empty_like(in_0, dtype=torch.float32)
    
    # Block size for softmax (use seq_len or max block size)
    BLOCK_SIZE = triton.next_power_of_2(seq_len)
    if BLOCK_SIZE > 1024:
        BLOCK_SIZE = 1024
    
    # Grid: (batch, heads, seq_len)
    grid = (batch_size, num_heads, seq_len)
    
    fused_attention_kernel[grid](
        in_0, in_1, out,
        batch_size, num_heads, seq_len,
        dropout_p, seed_offset,
        BLOCK_SIZE,
    )
    
    return out


def pattern(in_0, in_1):
    """
    Match the pattern: add + softmax + dropout + cast
    """
    tmp_0 = in_0 + in_1
    tmp_1 = torch.nn.functional.softmax(tmp_0, dim=-1)
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.1, False, False)
    tmp_3 = tmp_2.to(torch.float32)
    return tmp_3


def replacement_args(in_0, in_1):
    dropout_p = 0.1
    return (in_0, in_1, dropout_p)


def replacement_func():
    return fused_attention_wrapper