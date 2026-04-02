import torch
import triton
import triton.language as tl

def pattern(normalized_deg, row, col, edge_weight):
    """
    Fused sparse graph operations: indexing and multiplication
    This replaces: tmp_5 = normalized_deg[row]; tmp_6 = tmp_5 * edge_weight; 
                   tmp_7 = normalized_deg[col]; tmp_8 = tmp_6 * tmp_7
    Returns: tmp_8, tmp_5 (for backward compatibility if needed)
    """
    tmp_5 = normalized_deg[row]
    tmp_6 = tmp_5 * edge_weight
    tmp_7 = normalized_deg[col]
    tmp_8 = tmp_6 * tmp_7
    return tmp_8, tmp_5

def replacement_args(normalized_deg, row, col, edge_weight):
    return (normalized_deg, row, col, edge_weight)

@triton.jit
def fused_sparse_graph_kernel(
    deg_ptr,
    row_ptr,
    col_ptr,
    edge_weight_ptr,
    out_ptr,
    n_edges,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_edges
    
    # Load indices and weights
    row_idx = tl.load(row_ptr + offsets, mask=mask, other=0).to(tl.int32)
    col_idx = tl.load(col_ptr + offsets, mask=mask, other=0).to(tl.int32)
    edge_weight = tl.load(edge_weight_ptr + offsets, mask=mask, other=0.0)
    
    # Load normalized degree values
    max_vertex_idx = max(tl.max(row_idx), tl.max(col_idx)) + 1
    deg_row = tl.load(deg_ptr + row_idx, mask=tl.where(row_idx < max_vertex_idx, True, False), other=0.0)
    deg_col = tl.load(deg_ptr + col_idx, mask=tl.where(col_idx < max_vertex_idx, True, False), other=0.0)
    
    # Compute: edge_weight * deg_row * deg_col
    result = edge_weight * deg_row * deg_col
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap  
def fused_sparse_graph_ops(normalized_deg, row, col, edge_weight):
    n_edges = edge_weight.numel()
    BLOCK_SIZE = 1024  # Can be autotuned later
    
    # Ensure normalized_deg is on the same device and contiguous
    normalized_deg = normalized_deg.contiguous()
    row = row.contiguous()
    col = col.contiguous()
    edge_weight = edge_weight.contiguous()
    
    out = torch.empty_like(edge_weight)
    
    fused_sparse_graph_kernel[(n_edges + BLOCK_SIZE - 1) // BLOCK_SIZE, ](
        deg_ptr=normalized_deg,
        row_ptr=row,
        col_ptr=col,
        edge_weight_ptr=edge_weight,
        out_ptr=out,
        n_edges=n_edges,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out, normalized_deg[row]  # return out and tmp_5 for compatibility

def replacement_func():
    return fused_sparse_graph_ops