import torch
import triton
import triton.language as tl

def pattern(dividend, divisor, embedding_tensor):
    # Pattern: division + type conversion + addition
    # This pattern matches: tmp_4 = in_5 / in_4; tmp_5 = tmp_4.to(torch.float32); tmp_7 = tmp_5 + tmp_6
    # Note: We assume all tensors are already float32 (as per weight_meta), so .to(torch.float32) is redundant
    tmp_4 = dividend / divisor
    tmp_5 = tmp_4  # Skip redundant type conversion
    tmp_7 = tmp_5 + embedding_tensor  # Addition with embedding tensor
    return tmp_7

def replacement_args(dividend, divisor, embedding_tensor):
    return (dividend, divisor, embedding_tensor)

@triton.jit
def fused_kernel(
    dividend_ptr,
    divisor_ptr,
    embedding_ptr,
    output_ptr,
    batch_size,
    seq_len,
    hidden_dim,
    BLOCK_SIZE_HIDDEN: tl.constexpr,
):
    # Calculate program indices
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    hidden_idx = tl.program_id(2)
    
    # Compute global offsets
    batch_offset = batch_idx * seq_len * hidden_dim
    seq_offset = batch_idx * seq_len
    offset = batch_offset + seq_offset * hidden_dim + hidden_idx
    
    # Ensure we're within bounds for this thread
    if batch_idx >= batch_size or seq_idx >= seq_len or hidden_idx >= hidden_dim:
        return
    
    # Load values
    dividend_val = tl.load(dividend_ptr + offset)
    divisor_val = tl.load(divisor_ptr + seq_offset * hidden_dim + hidden_idx)
    embedding_val = tl.load(embedding_ptr + offset)
    
    # Fused computation: (dividend/divisor) + embedding
    # This avoids intermediate tensor allocations and redundant type conversions
    if divisor_val != 0:
        result = (dividend_val / divisor_val) + embedding_val
    else:
        result = dividend_val + embedding_val
    
    # Store result
    tl.store(output_ptr + offset, result)

@torch.fx.wrap
def fused_division_add_embedding(dividend, divisor, embedding):
    batch_size, seq_len, hidden_dim = dividend.shape
    
    # Output tensor
    output = torch.empty_like(dividend)
    
    # Optimize block size based on hidden dimension
    if hidden_dim <= 512:
        BLOCK_SIZE_HIDDEN = 64
    elif hidden_dim <= 1024:
        BLOCK_SIZE_HIDDEN = 128
    else:
        BLOCK_SIZE_HIDDEN = 256
    
    # Calculate grid size: (batch_size, seq_len, hidden_dim // BLOCK_SIZE_HIDDEN)
    grid = (
        batch_size,
        seq_len,
        (hidden_dim + BLOCK_SIZE_HIDDEN - 1) // BLOCK_SIZE_HIDDEN
    )
    
    # Launch kernel
    fused_kernel[grid](
        dividend_ptr=dividend,
        divisor_ptr=divisor,
        embedding_ptr=embedding,
        output_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        BLOCK_SIZE_HIDDEN=BLOCK_SIZE_HIDDEN,
    )
    
    return output

def replacement_func():
    return fused_division_add_embedding