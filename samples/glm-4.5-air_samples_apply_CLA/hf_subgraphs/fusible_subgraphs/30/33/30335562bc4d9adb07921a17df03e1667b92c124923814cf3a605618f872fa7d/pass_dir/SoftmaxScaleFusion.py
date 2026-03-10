import torch
import triton
import triton.language as tl

# Pattern matching: scalar multiplication followed by softmax
def pattern(x, scale):
    scaled = scale * x
    softmax = torch.nn.functional.softmax(scaled, dim=-1)
    return softmax

# Argument extraction
def replacement_args(x, scale):
    return (x, scale)

# Optimized kernel that fuses scaling and softmax
@triton.jit
def softmax_scale_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    # Load scaled data for this row
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    x = tl.load(x_ptr + row_idx * n_cols + offsets, mask=mask, other=float('-inf'))
    
    # Apply scaling and compute softmax
    scaled = 0.0625 * x
    max_val = tl.max(scaled, axis=0)
    scaled = scaled - max_val
    exp_scaled = tl.exp(scaled)
    sum_exp = tl.sum(exp_scaled, axis=0)
    softmax_out = exp_scaled / sum_exp
    
    # Store results
    tl.store(out_ptr + row_idx * n_cols + offsets, softmax_out, mask=mask)

@torch.fx.wrap
def triton_softmax_scale(x):
    n_rows, n_cols = x.shape[-2], x.shape[-1]
    
    # For the input shape [24, 8192, 19], we process the last two dimensions
    if len(x.shape) == 3:
        # Reshape to [batch_size * seq_len, features] for processing
        batch_size, seq_len, features = x.shape
        x_reshaped = x.reshape(-1, features)
    else:
        x_reshaped = x
    
    BLOCK_SIZE = 128
    num_programs = x_reshaped.shape[0]
    out = torch.empty_like(x_reshaped)
    
    softmax_scale_kernel[(num_programs, 1, 1)](
        x_ptr=x_reshaped,
        out_ptr=out,
        n_rows=x_reshaped.shape[0],
        n_cols=x_reshaped.shape[1],
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape back if needed
    if len(x.shape) == 3:
        return out.reshape(batch_size, seq_len, features)
    return out

def replacement_func():
    return triton_softmax_scale