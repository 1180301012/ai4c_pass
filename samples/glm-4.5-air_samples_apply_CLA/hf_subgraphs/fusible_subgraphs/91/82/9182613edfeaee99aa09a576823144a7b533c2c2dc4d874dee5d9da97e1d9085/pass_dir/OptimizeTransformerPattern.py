import torch
import triton
import triton.language as tl

def pattern(input_5d):
    """Pattern to match 5D to 4D reshape operation"""
    # Match: in_1.reshape(batch_size, 16, seq_len, head_dim)
    # The actual reshape is from [batch, 4, 4, seq_len, head_dim] to [batch, 16, seq_len, head_dim]
    batch_size, dim1, dim2, seq_len, head_dim = input_5d.shape
    reshaped_4d = input_5d.reshape(batch_size, 16, seq_len, head_dim)
    return reshaped_4d

def replacement_args(input_5d):
    """Extract arguments for the optimized reshape kernel"""
    batch_size, dim1, dim2, seq_len, head_dim = input_5d.shape
    return (input_5d, batch_size, seq_len, head_dim)

@triton.jit
def optimized_reshape_kernel(
    input_5d_ptr,
    output_4d_ptr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for 5D to 4D reshape with efficient memory layout conversion"""
    # Each program handles a contiguous block of data
    program_id = tl.program_id(0)
    block_start = program_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch_size * 16 * seq_len * head_dim)
    
    if tl.any(mask):
        # Calculate 4D coordinates from linear index
        idx_4d = offsets
        d4 = idx_4d % head_dim  # head_dim
        idx_4d = idx_4d // head_dim
        d3 = idx_4d % seq_len   # seq_len  
        idx_4d = idx_4d // seq_len
        d2_combined = idx_4d % 16  # Combined heads (4*4)
        b = idx_4d // 16          # batch
        
        # Calculate original 5D coordinates
        # [batch, 4, 4, seq_len, head_dim] -> [batch, 16, seq_len, head_dim]
        original_dim1 = d2_combined // 4  # Original dimension 1 (0-3)
        original_dim2 = d2_combined % 4   # Original dimension 2 (0-3)
        
        # Calculate input offset in 5D layout
        input_offset = ((b * 4 + original_dim1) * 4 + original_dim2) * (seq_len * head_dim) + (d3 * head_dim + d4)
        
        # Load from 5D layout and store to 4D layout
        input_val = tl.load(input_5d_ptr + input_offset, mask=mask, other=0.0)
        tl.store(output_4d_ptr + offsets, input_val, mask=mask)

@torch.fx.wrap
def optimized_reshape(input_5d):
    """Wrapper function for optimized 5D to 4D reshape"""
    batch_size, dim1, dim2, seq_len, head_dim = input_5d.shape
    
    # Create output tensor
    output_shape = (batch_size, 16, seq_len, head_dim)
    output = torch.empty(output_shape, dtype=input_5d.dtype, device=input_5d.device)
    
    # Calculate grid size
    total_elements = batch_size * 16 * seq_len * head_dim
    BLOCK_SIZE = 1024  # Optimal block size for GPU
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    optimized_reshape_kernel[(num_programs,)](
        input_5d_ptr=input_5d,
        output_4d_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        head_dim=head_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    """Return the optimized function"""
    return optimized_reshape