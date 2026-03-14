import torch
import triton
import triton.language as tl

def pattern(x, y):
    # x: tmp_7 (from embedding addition), [4, 512, 1280]
    # y: tmp_0 (attention_mask), [4, 512]
    
    tmp_8 = y.unsqueeze(-1)
    tmp_9 = x * tmp_8
    return tmp_9

def replacement_args(x, y):
    return (x, y)

@triton.jit
def optimized_broadcast_mul_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    num_sequences,
    seq_len,
    embed_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Total number of sequence positions
    total_positions = num_sequences * seq_len
    
    # Program identifier
    pid = tl.program_id(0)
    
    # Memory offsets for this program
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_positions
    
    # Load attention mask for positions in this block
    y_vals = tl.load(y_ptr + offsets, mask=mask, other=1.0)
    
    # Process each sequence position in the block
    for i in range(BLOCK_SIZE):
        if offsets[i] < total_positions:
            seq_idx = offsets[i] // seq_len
            pos_idx = offsets[i] % seq_len
            
            # Reshape mask value for broadcasting: [1] -> [embed_dim]
            mask_val = y_vals[i]
            
            # Load input values for this sequence position
            x_offset = (seq_idx * seq_len + pos_idx) * embed_dim + tl.arange(0, embed_dim)
            x_vals = tl.load(x_ptr + x_offset, mask=tl.arange(0, embed_dim) < embed_dim, other=0.0)
            
            # Perform multiplication with broadcasting
            out_vals = x_vals * mask_val
            
            # Store results
            tl.store(out_ptr + x_offset, out_vals, mask=tl.arange(0, embed_dim) < embed_dim)

@torch.fx.wrap
def optimized_broadcast_multiply(x, y):
    num_sequences, seq_len, embed_dim = x.shape
    
    # Flatten the attention mask
    y_flat = y.reshape(-1)  # [num_sequences * seq_len]
    
    # Use a conservative block size that works with power-of-2 constraints
    BLOCK_SIZE = 256  # Number of sequence positions per program
    
    # Calculate grid size
    total_positions = num_sequences * seq_len
    grid_size = (total_positions + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with same dtype as x
    out = torch.empty_like(x)
    
    # Launch the kernel with 1D grid
    optimized_broadcast_mul_kernel[grid_size](
        x_ptr=x,
        y_ptr=y_flat,
        out_ptr=out,
        num_sequences=num_sequences,
        seq_len=seq_len,
        embed_dim=embed_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_broadcast_multiply