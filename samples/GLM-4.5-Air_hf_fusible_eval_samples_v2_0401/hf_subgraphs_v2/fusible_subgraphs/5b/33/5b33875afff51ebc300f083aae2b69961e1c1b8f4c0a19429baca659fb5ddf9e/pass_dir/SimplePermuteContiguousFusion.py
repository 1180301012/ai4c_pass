import torch
import triton
import triton.language as tl

def pattern(x):
    """
    Match permute followed by contiguous operations
    Pattern: x.permute(0, 2, 1, 3).contiguous()
    Returns the contiguous result after permutation
    """
    permuted = x.permute(0, 2, 1, 3)
    contiguous_result = permuted.contiguous()
    return contiguous_result

def replacement_args(x):
    return (x,)

@triton.jit 
def simple_permute_kernel(
    input_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    dim1: tl.constexpr,
    dim2: tl.constexpr,
    dim3: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Simple kernel that fuses permutation (0,2,1,3) and contiguous memory layout
    Input shape: [batch_size, dim1, dim2, dim3]
    Output shape: [batch_size, dim2, dim1, dim3]
    """
    batch_idx = tl.program_id(0)
    elem_idx = tl.program_id(1)
    
    # Calculate sizes
    input_block_size = dim2 * dim3
    output_block_size = dim1 * dim3
    
    input_offset = batch_idx * input_block_size + elem_idx
    output_offset = batch_idx * output_block_size + elem_idx
    
    # Copy data with permuted indices
    for k in range(0, dim3, BLOCK_SIZE):
        mask = k + tl.arange(0, BLOCK_SIZE) < dim3
        
        # Get input coordinates: [batch, dim1, dim2, dim3]
        # Get corresponding output coordinates: [batch, dim2, dim1, dim3]
        
        # For the current element index, determine which dims to swap
        # elem_idx = (dim1 * dim2) index -> we need to split it
        local_dim2 = elem_idx // dim1  
        local_dim1 = elem_idx % dim1
        
        input_pos = batch_idx * (dim1 * dim2 * dim3) + (local_dim1 * dim2 + local_dim2) * dim3 + k
        output_pos = batch_idx * (dim2 * dim1 * dim3) + (local_dim2 * dim1 + local_dim1) * dim3 + k
        
        val = tl.load(input_ptr + input_pos, mask=mask, other=0.0)
        tl.store(output_ptr + output_pos, val, mask=mask)

@torch.fx.wrap
def optimized_permute_contiguous(x):
    """
    Optimized function that fuses permute(0,2,1,3) and contiguous operations
    Avoids creating intermediate tensor when possible
    """
    input_shape = x.shape
    
    # Only optimize for 4D tensors with specific pattern
    if len(input_shape) == 4:
        batch_size, dim1, dim2, dim3 = input_shape
        
        # Check if this matches the expected pattern from attention computation
        if dim3 <= 64:  # Only optimize for reasonable head dimensions
            output_shape = (batch_size, dim2, dim1, dim3)
            output = torch.empty(output_shape, dtype=x.dtype, device=x.device)
            
            BLOCK_SIZE = min(32, dim3)
            
            # Calculate grid dimensions
            total_elements = batch_size * dim1 * dim2
            grid = (batch_size, total_elements)
            
            simple_permute_kernel[grid](
                input_ptr=x,
                output_ptr=output,
                batch_size=batch_size,
                dim1=dim1,
                dim2=dim2,
                dim3=dim3,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            
            return output
    
    # Fallback to regular operations
    return x.permute(0, 2, 1, 3).contiguous()

def replacement_func():
    """Return optimized permute+contiguous fusion function"""
    return optimized_permute_contiguous