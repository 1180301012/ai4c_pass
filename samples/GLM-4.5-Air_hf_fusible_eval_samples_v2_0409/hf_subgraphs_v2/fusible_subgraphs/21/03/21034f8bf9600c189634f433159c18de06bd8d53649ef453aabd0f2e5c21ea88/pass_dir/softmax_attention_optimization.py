import torch
import triton
import triton.language as tl

@triton.jit
def optimized_max_kernel(
    x_ptr,
    max_ptr,
    batch_size,
    heads,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Each block processes one sequence for one batch
    batch_idx = pid // heads
    head_idx = pid % heads
    
    if batch_idx >= batch_size:
        return
        
    # Compute range for this sequence
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < seq_len
    
    # Load data
    x = tl.load(x_ptr + batch_idx * heads * seq_len + head_idx * seq_len + offsets,
                mask=mask, other=float('-inf'))
    
    # Compute max
    max_val = tl.max(x)
    
    # Store max
    tl.store(max_ptr + batch_idx * heads * seq_len + head_idx * seq_len + offsets[:1],
             max_val, mask=offsets[:1] < 1)

@triton.jit
def optimized_softmax_kernel(
    x_ptr,
    max_ptr,
    out_ptr,
    batch_size,
    heads,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Each block processes one sequence for one batch
    batch_idx = pid // heads
    head_idx = pid % heads
    
    if batch_idx >= batch_size:
        return
        
    # Compute range for this sequence
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < seq_len
    
    # Load data and max
    x = tl.load(x_ptr + batch_idx * heads * seq_len + head_idx * seq_len + offsets,
                mask=mask, other=float('-inf'))
    max_val = tl.load(max_ptr + batch_idx * heads * seq_len + head_idx * seq_len + offsets[:1])
    
    # Compute softmax
    exp_x = tl.exp(x - max_val)
    sum_exp = tl.sum(exp_x)
    softmax = exp_x / sum_exp
    
    # Store result
    tl.store(out_ptr + batch_idx * heads * seq_len + head_idx * seq_len + offsets,
             softmax, mask=mask)

@torch.fx.wrap
def optimized_softmax_2d(x):
    """
    Pure Triton softmax implementation without forbidden APIs
    """
    if x.dim() == 3:
        # [batch, heads, seq_len]
        n_batch, n_heads, seq_len = x.shape
    elif x.dim() == 4:
        # [batch, heads, seq_len1, seq_len2] -> reshape to 3D
        if x.shape[-1] == 1:
            # Remove last dimension
            x = x.squeeze(-1)
            n_batch, n_heads, seq_len = x.shape
        else:
            # Reshape to combine last two dimensions
            x = x.reshape(n_batch, n_heads, -1)
            n_batch, n_heads, seq_len = x.shape
    else:
        # For unsupported dimensions, raise error (the pass shouldn't match these cases)
        raise ValueError(f"Unsupported tensor dimensions for softmax: {x.dim()}")
    
    BLOCK_SIZE = 1024
    
    # Allocate intermediate arrays
    max_vals = torch.full((n_batch, n_heads, 1), float('-inf'), device=x.device, dtype=x.dtype)
    out = torch.empty_like(x)
    
    # Grid size for max computation
    max_grid = (n_batch * n_heads + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # First pass: compute max values
    optimized_max_kernel[max_grid](
        x,
        max_vals,
        n_batch, n_heads, seq_len,
        BLOCK_SIZE
    )
    
    # Broadcast max values to match sequence length
    max_vals_expanded = max_vals.expand(-1, -1, seq_len)
    
    # Grid size for softmax computation
    softmax_grid = (n_batch * n_heads + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Second pass: compute softmax
    optimized_softmax_kernel[softmax_grid](
        x,
        max_vals_expanded,
        out,
        n_batch, n_heads, seq_len,
        BLOCK_SIZE
    )
    
    return out



def pattern(x):
    return torch.nn.functional.softmax(x, dim=-1)

def replacement_args(x):
    return (x,)

def replacement_func():
    return optimized_softmax_2d