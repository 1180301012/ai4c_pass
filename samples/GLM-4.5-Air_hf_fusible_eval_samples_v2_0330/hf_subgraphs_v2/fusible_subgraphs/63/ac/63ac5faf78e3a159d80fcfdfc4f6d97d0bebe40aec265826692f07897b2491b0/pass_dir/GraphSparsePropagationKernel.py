import torch
import triton
import triton.language as tl

# Pattern matching for the graph sparse propagation operations
def pattern(deg, edge_weight, row, col):
    """
    Match the graph sparse propagation pattern:
    1. Degree normalization: deg.pow_(-0.5)
    2. Infinity handling: masked_fill with 0
    3. Sparse indexing: tmp_2[row] and tmp_2[col]
    4. Edge weight multiplication and final combination
    """
    # Degree normalization with infinity handling
    tmp_2 = deg.pow_(-0.5)
    tmp_3 = tmp_2.__eq__(float('inf'))
    tmp_4 = tmp_2.masked_fill_(tmp_3, 0.0)
    
    # Sparse indexing operations
    tmp_5 = tmp_4[row]  # row indexing
    tmp_6 = tmp_5 * edge_weight  # multiply by edge weights
    
    tmp_7 = tmp_4[col]  # col indexing
    tmp_8 = tmp_6 * tmp_7  # final combination
    
    return tmp_8, tmp_4  # tmp_4 is needed for intermediate observability

# Extract arguments for the replacement kernel
def replacement_args(deg, edge_weight, row, col):
    return (deg, edge_weight, row, col)

# Triton kernel for optimized graph sparse propagation
@triton.jit
def sparse_graph_kernel(
    deg_ptr,
    edge_weight_ptr,
    row_ptr,
    col_ptr,
    out_ptr,
    deg_size,
    edge_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    edge_start = pid * BLOCK_SIZE
    
    # Load edge indices and weights
    edge_offsets = edge_start + tl.arange(0, BLOCK_SIZE)
    edge_mask = edge_offsets < edge_size
    
    row_indices = tl.load(row_ptr + edge_offsets, mask=edge_mask, other=0)
    col_indices = tl.load(col_ptr + edge_offsets, mask=edge_mask, other=0)
    edge_weights = tl.load(edge_weight_ptr + edge_offsets, mask=edge_mask, other=0.0)
    
    # Load degree values and compute normalization
    deg_values = tl.load(deg_ptr + row_indices, mask=edge_mask, other=0.0)
    deg_norm = tl.where(deg_values == tl.inf, 0.0, deg_values)
    deg_norm = tl.rsqrt(deg_norm)  # pow(-0.5) equivalent
    
    deg_values_col = tl.load(deg_ptr + col_indices, mask=edge_mask, other=0.0)
    deg_norm_col = tl.where(deg_values_col == tl.inf, 0.0, deg_values_col)
    deg_norm_col = tl.rsqrt(deg_norm_col)
    
    # Compute output with fused operations
    result = deg_norm * edge_weights * deg_norm_col
    
    # Store results
    tl.store(out_ptr + edge_offsets, result, mask=edge_mask)

@torch.fx.wrap
def optimized_sparse_graph_propagation(deg, edge_weight, row, col):
    """
    Optimized version of the graph sparse propagation operations
    """
    edge_size = row.shape[0]
    BLOCK_SIZE = 1024  # Optimized block size for typical GPU architectures
    num_blocks = (edge_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(edge_weight, device=edge_weight.device)
    
    # Launch Triton kernel
    sparse_graph_kernel[(num_blocks,)](
        deg_ptr=deg,
        edge_weight_ptr=edge_weight,
        row_ptr=row,
        col_ptr=col,
        out_ptr=out,
        deg_size=deg.shape[0],
        edge_size=edge_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out, deg.pow_(-0.5).masked_fill_(deg.pow_(-0.5) == float('inf'), 0.0)

# Replacement function - must return a callable function reference
def replacement_func():
    return optimized_sparse_graph_propagation