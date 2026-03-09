import torch
import triton
import triton.language as tl

def pattern(x, y):
    # Pattern: multiplication with broadcasting (tmp_7 * tmp_8)
    # where tmp_7 has shape [batch, seq, hidden] and tmp_8 has shape [batch, seq, 1]
    result = x * y
    return result

def replacement_args(x, y):
    return (x, y)

@triton.jit
def broadcasting_multiply_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    batch_size,
    seq_len,
    hidden_size,
    batch_seq_size: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Parallel over batch and sequence dimensions
    batch_seq_idx = tl.program_id(0)
    k = tl.program_id(1)
    
    # Calculate batch and sequence indices
    batch_idx = batch_seq_idx // seq_len
    seq_idx = batch_seq_idx % seq_len
    
    # Check bounds
    batch_mask = batch_idx < batch_size
    seq_mask = seq_idx < seq_len
    k_mask = k < hidden_size
    mask = batch_mask & seq_mask & k_mask
    
    # Compute memory offsets
    x_offset = batch_idx * seq_len * hidden_size + seq_idx * hidden_size + k
    y_offset = batch_idx * seq_len + seq_idx  # y has shape [batch, seq, 1]
    out_offset = x_offset  # Same shape as x
    
    # Load values
    x_val = tl.load(x_ptr + x_offset, mask=mask, other=0.0)
    y_val = tl.load(y_ptr + y_offset, mask=mask, other=1.0)
    
    # Perform multiplication with broadcasting
    result = x_val * y_val
    
    # Store result
    tl.store(out_ptr + out_offset, result, mask=mask)

@torch.fx.wrap
def optimized_broadcasting_multiply(x, y):
    batch_size, seq_len, hidden_size = x.shape
    
    # Handle different y shapes that need broadcasting
    if y.dim() == 2:  # [batch, seq] - needs to be expanded to [batch, seq, 1]
        y = y.unsqueeze(-1)
    
    # Case 1: y has shape [batch, seq, 1] - most common case
    if y.shape == (batch_size, seq_len, 1):
        # Create efficient Triton kernel implementation
        total_batch_seq = batch_size * seq_len
        BLOCK_SIZE_K = 128  # Process 128 hidden dimensions at a time
        
        num_programs_bs = (total_batch_seq + 1)  # One program per batch-sequence pair
        num_programs_k = (hidden_size + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
        
        out = torch.empty_like(x)
        
        broadcasting_multiply_kernel[(num_programs_bs, num_programs_k)](
            x_ptr=x,
            y_ptr=y.squeeze(-1),  # Remove the last dim for easier indexing
            out_ptr=out,
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_size=hidden_size,
            batch_seq_size=total_batch_seq,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        )
        
        return out
    
    # Case 2: Use PyTorch broadcasting for other cases
    return x * y

def replacement_func():
    return optimized_broadcasting_multiply