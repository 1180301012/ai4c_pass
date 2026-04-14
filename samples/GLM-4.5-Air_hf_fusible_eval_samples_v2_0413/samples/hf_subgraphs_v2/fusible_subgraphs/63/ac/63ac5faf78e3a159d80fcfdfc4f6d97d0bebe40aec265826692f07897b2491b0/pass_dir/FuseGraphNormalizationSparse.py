import torch
import triton
import triton.language as tl
from torch import inf

# Simple pattern to match sparse indexing operations  
def pattern(norm_deg, row_idx, edge_wt, col_idx):
    # Get normalized degree values for sparse indices
    row_norm = norm_deg[row_idx]
    col_norm = norm_deg[col_idx]
    
    # Compute final normalized edge values
    result = row_norm * edge_wt * col_norm
    return result

# Argument extraction function - must match pattern signature
def replacement_args(norm_deg, row_idx, edge_wt, col_idx):
    return (norm_deg, row_idx, edge_wt, col_idx)

# Triton kernel for fused sparse indexing and multiplication with manual tuning
@triton.jit
def fused_sparse_kernel(
    norm_deg_ptr,      # normalized degree tensor pointer (CUDA)
    row_ptr,           # row indices pointer (CUDA) 
    edge_weight_ptr,   # edge weights pointer (CUDA)
    col_ptr,           # column indices pointer (CUDA)
    out_ptr,           # output pointer (CUDA)
    n_edges,           # number of edges
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_edges
    
    # Load normalized degree values for row and column indices
    row_norm = tl.load(norm_deg_ptr + tl.load(row_ptr + offsets, mask=mask), mask=mask, other=0.0)
    col_norm = tl.load(norm_deg_ptr + tl.load(col_ptr + offsets, mask=mask), mask=mask, other=0.0)
    
    # Load edge weights
    edge_weights = tl.load(edge_weight_ptr + offsets, mask=mask, other=0.0)
    
    # Fuse indexing and multiplication operations
    result = row_norm * edge_weights * col_norm
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap 
def fused_sparse_ops(norm_deg, row, edge_weight, col):
    """Fused kernel for sparse indexing and multiplication operations with autotuning"""
    n_edges = row.numel()
    BLOCK_SIZE = 512  # Smaller block size for better autotuning coverage
    num_programs = (n_edges + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Determine dtype and device from inputs
    dtype = norm_deg.dtype
    device = norm_deg.device
    
    # Create output tensor
    out = torch.empty(n_edges, dtype=dtype, device=device)
    
    # Launch kernel with autotuning
    if n_edges > 0:
        fused_sparse_kernel[(num_programs,)](
            norm_deg_ptr=norm_deg,
            row_ptr=row,
            edge_weight_ptr=edge_weight,
            col_ptr=col,
            out_ptr=out,
            n_edges=n_edges,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return out

# Replacement function - returns the fused kernel wrapper
def replacement_func():
    return fused_sparse_ops