import torch
import triton
import triton.language as tl

def pattern(in_2):
    """Match: transpose pattern only"""
    tmp_2 = in_2.transpose(1, 2)
    return tmp_2

def replacement_args(in_2):
    """Extract arguments for the replacement function"""
    return (in_2,)

@triton.jit
def optimized_transpose_kernel(
    input_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    num_heads: tl.constexpr,
    seq_len: tl.constexpr,
):
    """Optimized transpose kernel with vectorized operations
    
    Args:
        input_ptr: Pointer to input tensor [batch_size, num_heads, seq_len]  
        output_ptr: Pointer to output tensor [batch_size, seq_len, num_heads]
        batch_size: Number of batches (typically 1)
        num_heads: Number of heads (128)
        seq_len: Sequence length (19)
    """
    # Use 3D grid with vectorized processing
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    seq_idx = tl.program_id(2)
    
    # Out-of-bounds check (avoid chained OR operators)
    if batch_idx >= batch_size:
        return
    if head_idx >= num_heads:
        return
    if seq_idx >= seq_len:
        return
    
    # Process multiple elements at once for better efficiency
    # Load groups of 4 elements when possible for vectorization
    if seq_len >= 4:
        # Process 4 consecutive elements in sequence dimension
        offsets = seq_idx + tl.arange(0, 4)
        mask = offsets < seq_len
        
        # Load and transpose 4 elements at once
        input_vals = tl.load(input_ptr + batch_idx * num_heads * seq_len + head_idx * seq_len + offsets, mask=mask)
        output_offsets = batch_idx * seq_len * num_heads + offsets * num_heads + head_idx
        tl.store(output_ptr + output_offsets, input_vals, mask=mask)
    else:
        # Fallback for small sequence lengths
        input_offset = batch_idx * num_heads * seq_len + head_idx * seq_len + seq_idx
        output_offset = batch_idx * seq_len * num_heads + seq_idx * num_heads + head_idx
        
        input_val = tl.load(input_ptr + input_offset)
        tl.store(output_ptr + output_offset, input_val)

@torch.fx.wrap
def optimized_transpose(input_tensor):
    """Wrapper function with specialized optimization for small tensors"""
    # Get tensor dimensions
    batch_size, num_heads, seq_len = input_tensor.shape
    total_elements = batch_size * num_heads * seq_len
    
    # For very small tensors, try a different strategy
    if total_elements <= 4096:
        # Use a more optimized approach for small tensors
        # Launch fewer, larger blocks to reduce overhead
        BLOCK_SIZE = 32  # Even larger blocks for small problems
        
        grid_x = (batch_size + BLOCK_SIZE - 1) // BLOCK_SIZE
        grid_y = (num_heads + BLOCK_SIZE - 1) // BLOCK_SIZE
        grid_z = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        output = torch.empty((batch_size, seq_len, num_heads), dtype=input_tensor.dtype, device=input_tensor.device)
        
        # Use optimized kernel
        optimized_transpose_kernel[(grid_x, grid_y, grid_z)](
            input_ptr=input_tensor,
            output_ptr=output,
            batch_size=batch_size,
            num_heads=num_heads,
            seq_len=seq_len,
        )
    else:
        # Standard approach for larger tensors
        BLOCK_SIZE = 8
        grid_x = (batch_size + BLOCK_SIZE - 1) // BLOCK_SIZE
        grid_y = (num_heads + BLOCK_SIZE - 1) // BLOCK_SIZE  
        grid_z = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        output = torch.empty((batch_size, seq_len, num_heads), dtype=input_tensor.dtype, device=input_tensor.device)
        
        optimized_transpose_kernel[(grid_x, grid_y, grid_z)](
            input_ptr=input_tensor,
            output_ptr=output,
            batch_size=batch_size,
            num_heads=num_heads,
            seq_len=seq_len,
        )
    
    return output

def replacement_func():
    """Return the optimized kernel function"""
    return optimized_transpose