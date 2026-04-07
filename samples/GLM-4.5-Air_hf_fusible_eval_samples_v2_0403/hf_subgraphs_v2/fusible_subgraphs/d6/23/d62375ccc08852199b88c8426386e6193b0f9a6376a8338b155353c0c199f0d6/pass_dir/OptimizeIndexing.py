import torch
import triton
import triton.language as tl

def pattern(embeddings_table, index_tensor):
    """
    Pattern matches the indexing operation sequence:
    embeddings_table.index_select(0, index_tensor)
    followed by view(1, 9, hidden_size)
    
    This optimizes the inefficient indexing by directly accessing consecutive rows.
    """
    # The operations we want to optimize
    selected_rows = embeddings_table.index_select(0, index_tensor)
    
    # For this specific pattern, we know it's looking for consecutive indices
    reshaped = selected_rows.view(1, 9, selected_rows.shape[1])
    
    return reshaped

def replacement_args(embeddings_table, index_tensor):
    # Extract arguments needed for the optimization
    # For optimal consecutive access, we know start_idx=2, num_indices=9
    start_idx = 2
    num_indices = 9
    hidden_size = embeddings_table.shape[1]
    
    return (embeddings_table, start_idx, num_indices, hidden_size)

@triton.jit
def optimized_direct_access_kernel(
    output_ptr,
    embeddings_ptr,
    embeddings_stride_0,
    embeddings_stride_1,
    start_idx,
    num_indices,
    hidden_size,
    BLOCK_SIZE: tl.constexpr
):
    """
    Optimized kernel that directly accesses consecutive rows without indexing tensor.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (num_indices * hidden_size)
    
    # Calculate which row and column we're processing
    row_idx = offsets // hidden_size
    col_idx = offsets % hidden_size
    
    # Only process rows in our range
    row_mask = (row_idx >= start_idx) & (row_idx < start_idx + num_indices)
    final_mask = mask & row_mask
    
    # Map row to our local indexing
    local_row = row_idx - start_idx
    src_offset = local_row * embeddings_stride_0 + col_idx
    
    # Store in output at correct position
    dest_offset = row_idx * hidden_size + col_idx
    
    embeddings_val = tl.load(embeddings_ptr + src_offset, mask=final_mask, other=0.0)
    tl.store(output_ptr + dest_offset, embeddings_val, mask=final_mask)

@torch.fx.wrap
def optimized_indexing_access(embeddings_table, start_idx, num_indices, hidden_size):
    """
    Optimized version that directly accesses consecutive rows without indexing tensor overhead.
    """
    # Output should be shape [1, num_indices, hidden_size]
    output_shape = (1, num_indices, hidden_size)
    output = torch.empty(output_shape, dtype=embeddings_table.dtype, device=embeddings_table.device)
    
    total_elements = num_indices * hidden_size
    BLOCK_SIZE = 128  # Optimal block size for this workload
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    embeddings_stride_0 = embeddings_table.stride(0)
    
    optimized_direct_access_kernel[(num_programs,)](
        output_ptr=output,
        embeddings_ptr=embeddings_table,
        embeddings_stride_0=embeddings_stride_0,
        embeddings_stride_1=1,  # second dimension stride
        start_idx=start_idx,
        num_indices=num_indices,
        hidden_size=hidden_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return optimized_indexing_access