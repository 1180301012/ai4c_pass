import torch
import triton
import triton.language as tl


@triton.jit
def index_select_kernel(
    in_ptr,
    indices_ptr,
    out_ptr,
    num_indices: tl.constexpr,
    num_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized index_select kernel for gathering rows from a tensor.
    
    Args:
        in_ptr: Input tensor [N, num_cols]
        indices_ptr: Indices tensor [num_indices]
        out_ptr: Output tensor [num_indices, num_cols]
        num_indices: Number of indices
        num_cols: Number of columns in input/output
    """
    # Program handles a contiguous block of output rows
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Offsets for loading indices
    index_offsets = block_start + tl.arange(0, BLOCK_SIZE)
    index_mask = index_offsets < num_indices
    
    # Load indices
    indices = tl.load(indices_ptr + index_offsets, mask=index_mask, other=0)
    
    # For each column, compute source row and load data
    # Each thread block processes multiple output rows
    for col_offset in range(0, num_cols, 1):
        # Compute source row indices for all rows in this block
        row_indices = indices
        
        # Load from input: [num_indices, num_cols] output row-by-row
        for row_idx in range(BLOCK_SIZE):
            global_row_idx = block_start + row_idx
            if global_row_idx < num_indices:
                src_row = indices[row_idx]
                # Load from in[src_row, col_offset]
                src_offset = src_row * num_cols + col_offset
                val = tl.load(in_ptr + src_offset)
                # Store to out[row_idx, col_offset] within this block
                dst_offset = global_row_idx * num_cols + col_offset
                tl.store(out_ptr + dst_offset, val)


@torch.fx.wrap
def triton_index_select(in_0, in_1):
    """
    Optimized index_select implementation using Triton.
    in_0 is a tuple/list of tensors: (indices_tensor, other_tensor)
    We extract indices from in_0[0] and use in_1 as the source tensor.
    """
    # in_0 is a tuple containing [indices, other]
    # We need to use in_0[0] as the indices for index_select on in_1
    indices = in_0[0] if isinstance(in_0, (list, tuple)) else in_0
    
    # index_select along last dimension (-2 means second from last)
    # For 2D tensor [N, M], -2 = 0 (first dimension)
    num_indices = indices.shape[0]
    num_rows = in_1.shape[0]
    num_cols = in_1.shape[1]
    
    # Allocate output
    output = torch.empty((num_indices, num_cols), dtype=in_1.dtype, device=in_1.device)
    
    # Launch kernel with grid = num_indices
    BLOCK_SIZE = 128
    num_programs = num_indices
    
    index_select_kernel[(num_programs,)](
        in_ptr=in_1,
        indices_ptr=indices,
        out_ptr=output,
        num_indices=num_indices,
        num_cols=num_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def pattern(in_0, in_1):
    """Match the index_select pattern from GAE subgraph."""
    tmp_0 = in_0[1]
    tmp_1 = in_0[0]
    tmp_2 = in_1.index_select(-2, tmp_1)
    return (tmp_0, tmp_2)


def replacement_args(in_0, in_1):
    """Extract arguments needed for the optimized implementation."""
    return (in_0, in_1)


def replacement_func():
    """Return the optimized implementation."""
    return triton_index_select