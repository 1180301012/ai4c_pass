import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    # Try a very simple pattern first - just the addition operations
    tmp_0 = in_0 + in_3
    tmp_1 = tmp_0 + in_2
    tmp_2 = tmp_1 / 8.0
    tmp_3 = tmp_2 + in_1
    return tmp_3

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def simple_add_kernel_2d(
    in_0_ptr, in_1_ptr, in_2_ptr, in_3_ptr,
    out_ptr,
    batch_size: tl.constexpr,
    num_heads: tl.constexpr,
    seq_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Simplified 2D kernel approach with better memory access patterns
    pid = tl.program_id(0)
    total_elements = batch_size * num_heads * seq_len * seq_len
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Load inputs using safe flat indexing with proper broadcasting handling
    in_0 = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    # in_1 has shape [batch, 1, 1, seq_len] - for this specific shape, it only varies along seq_len dimension
    # Extract the batch and position, then map to in_1's 1D layout [batch * seq_len]
    batch_idx = offsets // (num_heads * seq_len * seq_len)
    position_idx = (offsets % (seq_len * seq_len)) // seq_len  # position within current sequence
    in_1_flat_offset = batch_idx * seq_len + position_idx
    in_1 = tl.load(in_1_ptr + in_1_flat_offset, mask=mask, other=0.0)
    in_2 = tl.load(in_2_ptr + offsets, mask=mask, other=0.0)
    in_3 = tl.load(in_3_ptr + offsets, mask=mask, other=0.0)
    
    # Compute fused operations: (in_0 + in_1 + in_2 + in_3) / 8.0
    result = (in_0 + in_1 + in_2 + in_3) / 8.0
    
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_simple_addition(in_0, in_1, in_2, in_3):
    batch_size, num_heads, seq_len, _ = in_0.shape
    
    # Use optimized block sizes based on sequence length for better performance
    BLOCK_SIZE = 256 if seq_len >= 512 else 128 if seq_len >= 128 else 64  # Even larger blocks for bigger sequences
    
    # Calculate grid dimensions - single dimension for simplicity
    total_elements = batch_size * num_heads * seq_len * seq_len
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (grid_size,)
    
    output = torch.empty_like(in_0)
    
    # Flatten inputs for easier pointer management
    in_0_flat = in_0.flatten()
    in_1_flat = in_1.flatten()
    in_2_flat = in_2.flatten()
    in_3_flat = in_3.flatten()
    output_flat = output.flatten()
    
    simple_add_kernel_2d[grid](
        in_0_ptr=in_0_flat,
        in_1_ptr=in_1_flat,
        in_2_ptr=in_2_flat,
        in_3_ptr=in_3_flat,
        out_ptr=output_flat,
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_simple_addition