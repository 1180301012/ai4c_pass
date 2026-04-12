import torch
import triton
import triton.language as tl
from torch import inf

# Pattern matching for just the sparse indexing operations
def pattern(deg_norm, row, edge_weight, col):
    # Perform sparse matrix multiplication with indexing
    row_vals = deg_norm[row]
    weighted_vals = row_vals * edge_weight
    col_vals = deg_norm[col]
    result = weighted_vals * col_vals
    
    return result

# Argument extraction function
def replacement_args(deg_norm, row, edge_weight, col):
    return (deg_norm, row, edge_weight, col)

# Triton kernel for optimized sparse matrix multiplication with degree normalization
@triton.jit
def graph_norm_spmm_kernel(
    deg_ptr, 
    row_ptr, 
    edge_weight_ptr,
    col_ptr,
    output_ptr,
    n_edges,
    BLOCK_SIZE: tl.constexpr,
    n_nodes: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_edges
    
    # Load indices and edge weights
    row_indices = tl.load(row_ptr + offsets, mask=mask, other=0)
    col_indices = tl.load(col_ptr + offsets, mask=mask, other=0)
    edge_weights = tl.load(edge_weight_ptr + offsets, mask=mask, other=0.0)
    
    # Load degree values and compute inverse square root
    deg_row_vals = tl.load(deg_ptr + row_indices, mask=tl.where(row_indices < n_nodes, True, False), other=0.0)
    deg_col_vals = tl.load(deg_ptr + col_indices, mask=tl.where(col_indices < n_nodes, True, False), other=0.0)
    
    # Compute inverse square root with proper dtype handling
    # Convert to fp32 for accurate computation
    deg_row_vals_fp32 = deg_row_vals.to(tl.float32)
    deg_col_vals_fp32 = deg_col_vals.to(tl.float32)
    
    # Check for zero degree values (which produce inf in rsqrt)
    deg_row_is_zero = deg_row_vals_fp32 == 0.0
    deg_col_is_zero = deg_col_vals_fp32 == 0.0
    
    # Compute inverse square root, handling zeros directly
    deg_row_inv_sqrt = tl.where(deg_row_is_zero, 0.0, tl.math.rsqrt(deg_row_vals_fp32))
    deg_col_inv_sqrt = tl.where(deg_col_is_zero, 0.0, tl.math.rsqrt(deg_col_vals_fp32)) 
    deg_row_norm = deg_row_inv_sqrt
    deg_col_norm = deg_col_inv_sqrt
    
    # Convert back to original dtype (supports both float16 and bfloat16)
    if deg_row_vals.dtype == tl.float16:
        deg_row_norm = deg_row_norm.to(tl.float16)
        deg_col_norm = deg_col_norm.to(tl.float16)
    else:
        deg_row_norm = deg_row_norm.to(tl.bfloat16)
        deg_col_norm = deg_col_norm.to(tl.bfloat16)
    
    # Perform sparse matrix multiplication
    weighted_vals = deg_row_norm * edge_weights
    result = weighted_vals * deg_col_norm
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def optimized_graph_norm_spmm(deg, row, edge_weight, col):
    n_edges = row.shape[0]
    n_nodes = deg.shape[0]
    BLOCK_SIZE = 1024
    num_programs = (n_edges + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Output shape matches the number of edges
    output = torch.empty(n_edges, dtype=deg.dtype, device=deg.device)
    
    graph_norm_spmm_kernel[(num_programs,)](
        deg_ptr=deg,
        row_ptr=row,
        edge_weight_ptr=edge_weight,
        col_ptr=col,
        output_ptr=output,
        n_edges=n_edges,
        BLOCK_SIZE=BLOCK_SIZE,
        n_nodes=n_nodes,
    )
    
    return output

# Replacement function
def replacement_func():
    return optimized_graph_norm_spmm