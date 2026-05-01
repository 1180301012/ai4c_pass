import torch
import triton
import triton.language as tl

def pattern(deg, row, col, edge_weight):
    # Compute tmp_2 = deg.pow(-0.5)
    tmp_2 = deg.pow(-0.5)
    # Handle inf: replace inf with 0
    tmp_3 = tmp_2.eq(float('inf'))
    tmp_4 = tmp_2.masked_fill(tmp_3, 0)
    # Get row-based normalization
    tmp_5 = tmp_4[row]
    # Multiply by edge_weight
    tmp_6 = tmp_5 * edge_weight
    # Get col-based normalization
    tmp_7 = tmp_4[col]
    # Final product
    tmp_8 = tmp_6 * tmp_7
    return tmp_8

def replacement_args(deg, row, col, edge_weight):
    return (deg, row, col, edge_weight)

@triton.jit
def graph_norm_kernel(
    deg_ptr,       # [deg_size]
    row_ptr,       # [num_edges]
    col_ptr,       # [num_edges]
    edge_weight_ptr, # [num_edges]
    out_ptr,       # [num_edges]
    deg_size,      # size of deg tensor
    num_edges,     # size of row, col, edge_weight
    BLOCK_SIZE: tl.constexpr,
):
    start_idx = tl.program_id(0) * BLOCK_SIZE
    end_idx = min(start_idx + BLOCK_SIZE, num_edges)
    
    for i in range(start_idx, end_idx):
        # Get index from row
        row_idx = tl.load(row_ptr + i)
        deg_val = tl.load(deg_ptr + row_idx)
        norm_row = tl.where(deg_val == 0.0, 0.0, tl.rsqrt(deg_val))
        
        # Get index from col
        col_idx = tl.load(col_ptr + i)
        deg_val2 = tl.load(deg_ptr + col_idx)
        norm_col = tl.where(deg_val2 == 0.0, 0.0, tl.rsqrt(deg_val2))
        
        # Multiply by edge weight
        edge_w = tl.load(edge_weight_ptr + i)
        out_val = norm_row * edge_w * norm_col
        tl.store(out_ptr + i, out_val)

@torch.fx.wrap
def graph_norm(deg, row, col, edge_weight):
    num_edges = row.size(0)
    deg_size = deg.size(0)
    
    # Allocate output
    out = torch.empty(num_edges, dtype=deg.dtype, device=deg.device)
    
    # Block size for Triton
    BLOCK_SIZE = 256
    num_blocks = (num_edges + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    graph_norm_kernel[(num_blocks,)](
        deg,  # deg_ptr
        row,  # row_ptr
        col,  # col_ptr
        edge_weight,  # edge_weight_ptr
        out,  # out_ptr
        deg_size,
        num_edges,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return graph_norm