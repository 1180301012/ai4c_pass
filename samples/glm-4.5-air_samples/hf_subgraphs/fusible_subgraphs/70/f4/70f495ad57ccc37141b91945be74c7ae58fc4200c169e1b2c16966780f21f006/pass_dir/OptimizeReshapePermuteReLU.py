import torch
import triton
import triton.language as tl
import math

def pattern(x, batch_size):
    # The indexing operation is essentially a no-op, so we skip it
    reshaped = x.reshape(batch_size, 16, 12, -1)
    permuted = reshaped.permute(0, 3, 1, 2)
    activated = torch.nn.functional.relu(permuted)
    return activated

def replacement_args(x, batch_size):
    return (x, batch_size)

@triton.jit
def reshape_permute_relu_kernel(
    x_ptr,
    out_ptr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    hidden_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size * seq_len * hidden_size
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate the new dimensions after reshape + permute
    # Original: (batch_size, seq_len, hidden_size)
    # After reshape: (batch_size, 16, 12, hidden_size/(16*12))
    # After permute: (batch_size, hidden_size/(16*12), 16, 12)
    new_hidden = hidden_size // (16 * 12)
    
    # Calculate original indices
    batch_idx = offsets // (seq_len * hidden_size)
    batch_idx = batch_idx % batch_size
    
    seq_idx = (offsets // hidden_size) % seq_len
    
    hidden_idx = offsets % hidden_size
    
    # Calculate new indices for the permuted layout
    # Map seq_len (192) to (16, 12) and permute to (hidden_size/(16*12), 16, 12)
    new_hidden_idx = hidden_idx // (16 * 12)
    seq_row_idx = (hidden_idx // 12) % 16
    seq_col_idx = hidden_idx % 12
    
    # Calculate output index in the final layout: (batch_size, new_hidden, 16, 12)
    output_idx = batch_idx * (new_hidden * 16 * 12) + new_hidden_idx * (16 * 12) + seq_row_idx * 12 + seq_col_idx
    
    # Apply ReLU activation
    activated = tl.math.maximum(x, 0.0)
    
    # Store result
    tl.store(out_ptr + output_idx, activated, mask=mask)

@torch.fx.wrap
def optimized_reshape_permute_relu(x, batch_size):
    # Get input dimensions
    batch_size_in, seq_len, hidden_size = x.shape
    
    # Calculate total elements and set up kernel launch
    n_elements = batch_size_in * seq_len * hidden_size
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with the correct final shape
    new_hidden = hidden_size // (16 * 12)
    out_shape = (batch_size_in, new_hidden, 16, 12)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Launch kernel
    reshape_permute_relu_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        batch_size=batch_size_in,
        seq_len=seq_len,
        hidden_size=hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_reshape_permute_relu