import torch
import triton
import triton.language as tl

def pattern(x, weight, indices):
    # x: tmp_5 (from division), [4, 512, 1280]
    # weight: tmp_1 (embedding weights), [1002, 1280]
    # indices: in_6 (position_ids), [4, 512]
    
    tmp_6 = torch.nn.functional.embedding(indices, weight, 1, None, 2.0, False, False)
    tmp_7 = x + tmp_6
    return tmp_7

def replacement_args(x, weight, indices):
    return (x, weight, indices)

@triton.jit
def optimized_embedding_kernel(
    indices_ptr,
    weight_ptr,
    x_ptr,
    out_ptr,
    num_sequences,
    sequence_len,
    vocab_size,
    embed_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Total number of sequence positions
    total_positions = num_sequences * sequence_len
    
    # Program identifier
    pid = tl.program_id(0)
    
    # Memory offsets for this program
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_positions
    
    # Load indices (position IDs)
    indices = tl.load(indices_ptr + offsets, mask=mask, other=0)
    
    # Process each position in the block
    for i in range(BLOCK_SIZE):
        if offsets[i] < total_positions:
            seq_idx = offsets[i] // sequence_len
            pos_idx = offsets[i] % sequence_len
            
            # Get the embedding for this position
            embed_idx = indices[i]
            embed_offset = embed_idx * embed_dim
            
            # Load embedding lookup
            embed_vals = tl.load(weight_ptr + embed_offset + tl.arange(0, embed_dim), mask=tl.arange(0, embed_dim) < embed_dim, other=0.0)
            
            # Load input values
            x_offset = (seq_idx * sequence_len + pos_idx) * embed_dim + tl.arange(0, embed_dim)
            x_vals = tl.load(x_ptr + x_offset, mask=tl.arange(0, embed_dim) < embed_dim, other=0.0)
            
            # Perform addition
            out_vals = embed_vals + x_vals
            
            # Store results
            tl.store(out_ptr + x_offset, out_vals, mask=tl.arange(0, embed_dim) < embed_dim)

@torch.fx.wrap
def optimized_embedding_addition(x, weight, indices):
    num_sequences, sequence_len, embed_dim = x.shape
    vocab_size = weight.shape[0]
    
    # Flatten indices for easier processing
    flat_indices = indices.reshape(-1)
    
    # Use a conservative block size that works with power-of-2 constraints
    BLOCK_SIZE = 256  # Number of sequence positions per program
    
    # Calculate grid size
    total_positions = num_sequences * sequence_len
    grid_size = (total_positions + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch the kernel with 1D grid
    optimized_embedding_kernel[grid_size](
        indices_ptr=flat_indices,
        weight_ptr=weight,
        x_ptr=x,
        out_ptr=out,
        num_sequences=num_sequences,
        sequence_len=sequence_len,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_embedding_addition