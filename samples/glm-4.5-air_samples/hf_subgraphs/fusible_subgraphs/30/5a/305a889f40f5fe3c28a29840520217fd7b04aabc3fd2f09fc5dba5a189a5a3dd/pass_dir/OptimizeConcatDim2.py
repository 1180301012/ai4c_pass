import torch
import triton
import triton.language as tl

def pattern(in_1, tmp_6):
    tmp_7 = torch.cat([in_1, tmp_6], dim=2)
    return tmp_7

def replacement_args(in_1, tmp_6):
    return (in_1, tmp_6)

@triton.jit
def optimized_concat_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    x_batch,
    x_heads,
    x_dim1,
    x_features,
    y_dim1,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Calculate which chunk each offset corresponds to
    chunk_size = x_features * y_dim1
    chunk_idx = offsets // chunk_size
    offset_in_chunk = offsets % chunk_size
    
    # Determine if this is in the x (in_1) or y (tmp_6) section
    # Total chunks = x_batch * x_heads * (x_dim1 + y_dim1)
    # x section covers first x_dim1 chunks per "row", y section covers next y_dim1 chunks
    
    # For each position (batch, head), we have x_dim1 + y_dim1 chunks along dim1
    elements_per_row = x_dim1 + y_dim1
    row_idx = chunk_idx // elements_per_row
    pos_in_row = chunk_idx % elements_per_row
    
    # Determine source and relative position
    if pos_in_row < x_dim1:
        # Load from x (in_1)
        source_offset = row_idx * x_dim1 * x_features + pos_in_row * x_features + offset_in_chunk
        source_data = tl.load(x_ptr + source_offset, mask=mask, other=0.0)
    else:
        # Load from y (tmp_6) - adjust position in row index
        adj_pos_in_row = pos_in_row - x_dim1
        source_offset = (row_idx * x_heads + row_idx) * y_dim1 * x_features + adj_pos_in_row * x_features + offset_in_chunk
        source_data = tl.load(y_ptr + source_offset, mask=mask, other=0.0)
    
    # Store in output at original offset
    tl.store(out_ptr + offsets, source_data, mask=mask)

@torch.fx.wrap  
def optimized_concat_dim2(x, y):
    # Get shapes
    x_shape = x.shape  # Expected: [1, 6, 1, 64]
    y_shape = y.shape  # Expected: [1, 6, 256, 64] orSimilar
    
    batch, heads, x_dim1, features = x_shape
    y_dim1 = y_shape[2]
    
    # Output shape
    out_shape = [batch, heads, x_dim1 + y_dim1, features]
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    total_elements = batch * heads * (x_dim1 + y_dim1) * features
    
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_concat_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        x_batch=batch,
        x_heads=heads,
        x_dim1=x_dim1,
        x_features=features,
        y_dim1=y_dim1,
        n_elements=total_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_concat_dim2