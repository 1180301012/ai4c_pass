import torch
import triton
import triton.language as tl
import math

def pattern(in_3, in_5, in_4, in_2):
    # Match the exact computation sequence from the model
    # tmp_2 = in_3.pow_(-0.5)
    # tmp_3 = tmp_2.__eq__(inf)
    # tmp_4 = tmp_2.masked_fill_(tmp_3, 0)
    # tmp_5 = tmp_2[in_5]
    # tmp_6 = tmp_5 * in_4
    # tmp_7 = tmp_4[in_2]
    # tmp_8 = tmp_6 * tmp_7
    # Return the final result (tmp_8)
    
    tmp_2 = in_3.pow_(-0.5)
    tmp_3 = tmp_2.__eq__(float('inf'))
    tmp_4 = tmp_2.masked_fill_(tmp_3, 0.0)
    tmp_5 = tmp_4[in_5]
    tmp_6 = tmp_5 * in_4
    tmp_7 = tmp_4[in_2]
    tmp_8 = tmp_6 * tmp_7
    
    return tmp_8

def replacement_args(in_3, in_4, in_2, in_5):
    return (in_3, in_4, in_2, in_5)

@triton.jit
def graph_message_passing_kernel(
    deg_ptr,
    edge_weight_ptr,
    col_ptr,
    row_ptr,
    out_ptr,
    num_nodes,
    num_edges,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for graph message passing with degree normalization"""
    # Compute degree normalization factors
    pid = tl.program_id(0)
    deg_offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = deg_offset < num_nodes
    
    # Load degrees and compute inverse square root
    deg = tl.load(deg_ptr + deg_offset, mask=mask, other=0.0)
    norm_factors = tl.math.rsqrt(deg)
    norm_factors = tl.where(deg > 0, norm_factors, 0.0)
    
    # Store normalization factors
    tl.store(out_ptr + deg_offset, norm_factors, mask=mask)
    
    if num_edges > 0:
        # Process edges in parallel blocks
        edge_block_size = 256
        edge_blocks = (num_edges + edge_block_size - 1) // edge_block_size
        
        for block in range(edge_blocks):
            block_start = block * edge_block_size
            edge_idx = block_start + tl.arange(0, edge_block_size)
            edge_mask = edge_idx < num_edges
            
            # Load edge information
            row_idx = tl.load(row_ptr + edge_idx, mask=edge_mask, other=0)
            col_idx = tl.load(col_ptr + edge_idx, mask=edge_mask, other=0)
            edge_weight = tl.load(edge_weight_ptr + edge_idx, mask=edge_mask, other=0.0)
            
            # Load row and column normalization factors
            row_norm = tl.load(out_ptr + row_idx, mask=edge_mask, other=0.0)
            col_norm = tl.load(out_ptr + col_idx, mask=edge_mask, other=0.0)
            
            # Compute message: edge_weight * row_norm * col_norm
            message = edge_weight * row_norm * col_norm
            tl.store(out_ptr + edge_idx, message, mask=edge_mask)

@torch.fx.wrap
def optimized_graph_message_passing(deg, edge_weight, col, row):
    """Optimized implementation of graph message passing with degree normalization"""
    num_nodes = deg.shape[0]
    num_edges = edge_weight.shape[0] if edge_weight is not None else 0
    
    # Choose optimal block size
    BLOCK_SIZE = 1024
    
    # Allocate output buffer
    out = torch.empty(num_edges, dtype=deg.dtype, device=deg.device)
    
    if num_edges > 0:
        # Launch kernel for processing edges
        num_blocks = (num_edges + BLOCK_SIZE - 1) // BLOCK_SIZE
        graph_message_passing_kernel[(num_blocks,)](
            deg_ptr=deg,
            edge_weight_ptr=edge_weight,
            col_ptr=col,
            row_ptr=row,
            out_ptr=out,
            num_nodes=num_nodes,
            num_edges=num_edges,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return out

def replacement_func():
    return optimized_graph_message_passing