import torch
import triton
import triton.language as tl

def pattern(x):
    # Transpose back from attention format
    tmp_transpose = x.transpose(1, 2)
    # Reshape to final output format - avoid * unpacking
    tmp_reshape = tmp_transpose.reshape(x.shape[0], x.shape[1], x.shape[1] * x.shape[3])
    return tmp_reshape

def replacement_args(x):
    return (x,)

@triton.jit
def transpose_reshape_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    seq_len,
    num_heads,
    head_dim,
    output_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Each thread handles one element
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Calculate total elements
    total_elements = batch_size * seq_len * output_dim
    
    # Create mask
    mask = idx < total_elements
    
    # Calculate original indices (before reshape)
    # idx: element index in output [batch, seq, output_dim]
    batch_idx = idx // (seq_len * output_dim)
    seq_idx = (idx % (seq_len * output_dim)) // output_dim
    head_idx = (idx % output_dim) // head_dim
    dim_idx = idx % head_dim
    
    # Calculate new indices (after transpose + reshape)
    # Original format: [batch, seq, num_heads, head_dim] -> [batch, num_heads, seq, head_dim] -> [batch, seq, output_dim]
    new_idx = (batch_idx * seq_len * num_heads * head_dim + 
               head_idx * seq_len * head_dim + 
               seq_idx * head_dim + dim_idx)
    
    # Load data from input
    x_val = tl.load(x_ptr + new_idx, mask=mask, other=0.0)
    
    # Store to output with direct index mapping
    tl.store(out_ptr + idx, x_val, mask=mask)

@torch.fx.wrap
def transpose_reshape_fused(x):
    batch_size = x.shape[0]
    seq_len = x.shape[1]
    num_heads = x.shape[2]
    head_dim = x.shape[3]
    output_dim = seq_len * head_dim
    
    # Create output tensor
    out = torch.empty((batch_size, seq_len, output_dim), dtype=x.dtype, device=x.device)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    num_blocks = (batch_size * seq_len * output_dim + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    transpose_reshape_kernel[(num_blocks,)](
        x_ptr=x,
        out_ptr=out,
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        output_dim=output_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return transpose_reshape_fused