import torch
import triton
import triton.language as tl


@triton.jit
def fused_gnn_sparse_kernel(
    deg_ptr,
    edge_weight_ptr,
    row_ptr,
    col_ptr,
    out_ptr,
    n_edges: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of edges
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_edges
    
    # Load row and col indices (int64)
    row_idx = tl.load(row_ptr + offsets, mask=mask, other=0).to(tl.int32)
    col_idx = tl.load(col_ptr + offsets, mask=mask, other=0).to(tl.int32)
    
    # Load edge weights
    edge_weight = tl.load(edge_weight_ptr + offsets, mask=mask, other=0.0)
    
    # Load degrees and compute inverse square root with inf handling
    deg_row = tl.load(deg_ptr + row_idx, mask=mask, other=0.0)
    deg_col = tl.load(deg_ptr + col_idx, mask=mask, other=0.0)
    
    # Compute deg^(-0.5) using rsqrt
    deg_row_inv_sqrt = tl.rsqrt(deg_row)
    deg_col_inv_sqrt = tl.rsqrt(deg_col)
    
    # Handle inf values: if deg is 0 or inf, result should be 0
    # rsqrt(inf) = 0, rsqrt(0) = inf, so we need to mask these cases
    is_valid_row = (deg_row > 0) & (deg_row < 1e38)
    is_valid_col = (deg_col > 0) & (deg_col < 1e38)
    is_valid = is_valid_row & is_valid_col
    
    # Set invalid entries to 0
    deg_row_inv_sqrt = tl.where(is_valid, deg_row_inv_sqrt, 0.0)
    deg_col_inv_sqrt = tl.where(is_valid, deg_col_inv_sqrt, 0.0)
    
    # Multiply: edge_weight * deg[row]^(-0.5) * deg[col]^(-0.5)
    result = edge_weight * deg_row_inv_sqrt * deg_col_inv_sqrt
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Match the GNN sparse computation pattern (without linear):
    tmp_2 = in_3.pow_(-0.5)
    tmp_3 = tmp_2.__eq__(inf)
    tmp_4 = tmp_2.masked_fill_(tmp_3, 0)
    tmp_5 = tmp_2[in_5]
    tmp_6 = tmp_5 * in_4
    tmp_7 = tmp_2[in_2]
    tmp_8 = tmp_6 * tmp_7
    return tmp_8
    """
    tmp_2 = in_3.pow_(-0.5)
    tmp_3 = tmp_2.__eq__(float('inf'))
    tmp_4 = tmp_2.masked_fill_(tmp_3, 0)
    tmp_5 = tmp_4[in_5]
    tmp_6 = tmp_5 * in_4
    tmp_7 = tmp_4[in_2]
    tmp_8 = tmp_6 * tmp_7
    return tmp_8


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    # Extract arguments for the sparse computation: col, deg, edge_weight, row
    # in_0 and in_1 (linear weights) are unused in this pass
    return (in_2, in_3, in_4, in_5)


@torch.fx.wrap
def fused_gnn_sparse_wrapper(in_2, in_3, in_4, in_5):
    """
    Fused GNN sparse kernel that combines:
    - Inverse square root computation with inf handling
    - Indexing with row/col indices
    - Multiplication with edge weights
    """
    n_edges = in_2.shape[0]
    BLOCK_SIZE = 256
    num_programs = (n_edges + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Output tensor for the sparse computation
    out_sparse = torch.empty_like(in_4)
    
    # Launch the fused kernel
    fused_gnn_sparse_kernel[(num_programs,)](
        deg_ptr=in_3,
        edge_weight_ptr=in_4,
        row_ptr=in_5,
        col_ptr=in_2,
        out_ptr=out_sparse,
        n_edges=n_edges,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out_sparse


def replacement_func():
    return fused_gnn_sparse_wrapper