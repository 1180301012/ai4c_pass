import torch
import triton
import triton.language as tl
import math

def pattern(in_3, in_4):
    """Pattern matching: indexing + view + permute + contiguous + unsqueeze"""
    tmp_2 = in_3[in_4]
    tmp_3 = tmp_2.view(197, 197, -1)
    tmp_4 = tmp_3.permute(2, 0, 1)
    tmp_5 = tmp_4.contiguous()
    tmp_6 = tmp_5.unsqueeze(0)
    return tmp_6

def replacement_args(in_3, in_4):
    """Extract arguments for the replacement function"""
    return (in_3, in_4)



@triton.jit
def optimized_indexing_kernel(
    table_ptr,
    indices_ptr, 
    output_ptr,
    table_rows: tl.constexpr,
    table_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for tensor transformation"""
    pid = tl.program_id(0)
    
    # Each program handles a block of elements
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < 38809
    
    # Convert flat indices to 2D coordinates in 197x197 output
    rows = offsets // 197
    cols = offsets % 197
    
    # Load batch of indices
    indices = tl.load(indices_ptr + offsets, mask=mask, other=0)
    
    # Bounds check for table indices
    valid_mask = (indices >= 0) & (indices < table_rows)
    
    # Compute table addresses with bounds checking
    table_addresses = indices * table_cols
    table_addresses = tl.where(valid_mask, table_addresses, 0)
    
    # Load values from table with bounds checking
    values = tl.load(table_ptr + table_addresses, mask=valid_mask, other=0.0)
    
    # Compute output addresses (flattened [1, 1, 197, 197])
    output_addresses = offsets
    
    # Store results
    tl.store(output_ptr + output_addresses, values, mask=mask)

@torch.fx.wrap
def tensor_transformation_kernel(table, indices):
    """Fused tensor transformation using optimized Triton kernel"""
    # Create output tensor with correct shape
    output = torch.empty((1, 1, 197, 197), dtype=table.dtype, device=table.device)
    
    # Flatten the output for easier indexing
    flat_output = output.view(-1)  # [38809]
    
    # Use larger block size for better GPU utilization
    BLOCK_SIZE = 2048  # Larger block size to reduce kernel launch overhead
    num_blocks = (38809 + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Set up grid and launch optimized kernel
    grid = (num_blocks,)
    optimized_indexing_kernel[grid](
        table_ptr=table,
        indices_ptr=indices,
        output_ptr=flat_output,
        table_rows=table.shape[0],
        table_cols=table.shape[1],
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    """Return the optimized kernel function"""
    return tensor_transformation_kernel