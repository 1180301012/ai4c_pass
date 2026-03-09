import torch
import triton
import triton.language as tl

def pattern(x):
    # Match the normalization sequence from the original model:
    tmp_0 = x.sum(dim=-1)
    tmp_1 = tmp_0.unsqueeze(-1)
    # Using regular division instead of in-place
    result = x / tmp_1
    return result

def replacement_args(x):
    return (x,)

@triton.jit
def row_normalization_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    num_heads,
    seq_len,
    feat_dim,
    BLOCK_SIZE_M: tl.constexpr,
):
    # Simple grid setup - each program handles one element
    pid = tl.program_id(0)
    
    # Calculate which element this program handles
    batch = pid // (num_heads * seq_len * feat_dim)
    remainder = pid % (num_heads * seq_len * feat_dim)
    
    head = remainder // (seq_len * feat_dim)  
    remainder = remainder % (seq_len * feat_dim)
    
    seq = remainder // feat_dim
    feat = remainder % feat_dim
    
    # Calculate row base address (start of current row)
    row_base = batch * (num_heads * seq_len * feat_dim) + head * (seq_len * feat_dim) + seq * feat_dim
    
    # Load entire row for sum calculation
    row_offsets = row_base + tl.arange(0, feat_dim)
    row_mask = tl.arange(0, feat_dim) < feat_dim
    
    row_data = tl.load(x_ptr + row_offsets, mask=row_mask, other=0.0)
    row_sum = tl.sum(row_data)
    
    # Handle division by zero
    row_sum_safe = tl.where(row_sum == 0, tl.float32(1e-8), row_sum)
    
    # Load current element and normalize
    x_val = tl.load(x_ptr + pid, mask=pid < (batch_size * num_heads * seq_len * feat_dim), other=0.0)
    out_val = x_val / row_sum_safe
    
    # Store result
    tl.store(out_ptr + pid, out_val, mask=pid < (batch_size * num_heads * seq_len * feat_dim))

@torch.fx.wrap
def triton_row_normalization(x):
    batch, heads, seq, features = x.shape
    
    out = torch.empty_like(x)
    total_elements = batch * heads * seq * features
    block_size = 1024
    grid = (total_elements + block_size - 1) // block_size
    
    row_normalization_kernel[grid](
        x, out, batch, heads, seq, features,
        BLOCK_SIZE_M=128
    )
    
    return out

def replacement_func():
    return triton_row_normalization