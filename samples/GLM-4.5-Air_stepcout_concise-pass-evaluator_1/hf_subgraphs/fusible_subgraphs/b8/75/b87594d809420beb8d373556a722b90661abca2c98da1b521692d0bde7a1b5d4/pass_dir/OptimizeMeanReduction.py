import torch
import triton
import triton.language as tl

def pattern(in_3):
    """Optimize in_3.mean(-2) operation"""
    tmp_3 = in_3.mean(-2)
    return tmp_3

def replacement_args(in_3):
    return (in_3,)

@triton.jit
def mean_kernel(
    input_ptr,  # [batch_size, seq_len, feature_dim]
    output_ptr, # [batch_size, feature_dim]
    batch_size,
    seq_len,
    feature_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one feature in one batch
    batch = tl.program_id(0) // feature_dim
    feature = tl.program_id(0) % feature_dim
    
    # Initialize accumulator
    acc = 0.0
    
    # Compute number of blocks needed
    num_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Mask to track if we've processed any valid elements
    any_valid = tl.zeros((), dtype=tl.int1)
    
    for block in range(num_blocks):
        # Load a block of sequence elements for this batch and feature
        seq_offset = block * BLOCK_SIZE
        seq_indices = seq_offset + tl.arange(0, BLOCK_SIZE)
        mask = seq_indices < seq_len
        
        # Load data
        input_offset = batch * seq_len * feature_dim + feature + seq_indices * feature_dim
        vals = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
        
        # Accumulate
        acc += tl.sum(vals)
        if tl.any(mask):
            any_valid = 1
    
    # Compute mean
    mean_val = acc / seq_len
    
    # Store result
    output_offset = batch * feature_dim + feature
    tl.store(output_ptr + output_offset, mean_val)

@torch.fx.wrap
def triton_mean(x):
    batch_size, seq_len, feature_dim = x.shape
    
    out = torch.empty((batch_size, feature_dim), dtype=torch.float32, device=x.device)
    
    # Block size for tiling across sequence dimension
    BLOCK_SIZE = 128
    
    # Calculate grid size - each CUDA block handles one feature in one batch
    grid_size = batch_size * feature_dim
    
    mean_kernel[grid_size,](
        input_ptr=x,
        output_ptr=out,
        batch_size=batch_size,
        seq_len=seq_len,
        feature_dim=feature_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return triton_mean