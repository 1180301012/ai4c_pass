import torch
import triton
import triton.language as tl
from torch import inf

# Pattern for RECT_L-style where linear(in_1, in_0) with in_0=weight, in_1=input
def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    tmp_2 = in_3.pow_(-0.5)
    tmp_3 = tmp_2.__eq__(inf)
    tmp_4 = tmp_2.masked_fill_(tmp_3, 0)
    tmp_5 = tmp_2[in_5]
    tmp_6 = tmp_5 * in_4
    tmp_7 = tmp_2[in_2]
    tmp_8 = tmp_6 * tmp_7
    linear = torch.nn.functional.linear(in_1, in_0, None)
    return (tmp_8, linear)

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5, "route_10")

# Triton kernel: fused normalization (deg^(-0.5) with 0 for deg=0)
@triton.jit
def gnn_norm_kernel_10(
    deg_ptr, row_ptr, col_ptr, edge_weight_ptr, out_ptr,
    num_nodes, num_edges,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask_nodes = offsets < num_nodes

    deg = tl.load(deg_ptr + offsets, mask=mask_nodes, other=0.0)
    norm = tl.where(deg == 0.0, 0.0, tl.math.rsqrt(deg))
    norm = tl.where(norm > 1e30, 0.0, norm)
    norm = tl.where(norm != norm, 0.0, norm)

# Triton kernel: compute normalized edge weights
@triton.jit
def gnn_edge_norm_kernel_10(
    norm_ptr, row_ptr, col_ptr, edge_weight_ptr, out_ptr,
    num_edges,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask_edges = offsets < num_edges

    row_idx = tl.load(row_ptr + offsets, mask=mask_edges, other=0)
    col_idx = tl.load(col_ptr + offsets, mask=mask_edges, other=0)
    ew = tl.load(edge_weight_ptr + offsets, mask=mask_edges, other=0.0)

    norm_row = tl.load(norm_ptr + row_idx)
    norm_col = tl.load(norm_ptr + col_idx)

    out_val = norm_row * ew * norm_col
    tl.store(out_ptr + offsets, out_val, mask=mask_edges)

@torch.fx.wrap
def gnn_norm_fused_10(in_3, in_2, in_4, in_5):
    num_nodes = in_3.shape[0]
    num_edges = in_4.shape[0]
    dtype = in_3.dtype

    BLOCK_SIZE_NODES = 256
    num_programs_nodes = (num_nodes + BLOCK_SIZE_NODES - 1) // BLOCK_SIZE_NODES
    norm = torch.empty(num_nodes, dtype=dtype, device=in_3.device)

    gnn_norm_kernel_10[(num_programs_nodes,)](
        deg_ptr=in_3, row_ptr=in_5, col_ptr=in_2, edge_weight_ptr=in_4, out_ptr=norm,
        num_nodes=num_nodes, num_edges=num_edges,
        BLOCK_SIZE=BLOCK_SIZE_NODES,
    )

    BLOCK_SIZE_EDGES = 256
    num_programs_edges = (num_edges + BLOCK_SIZE_EDGES - 1) // BLOCK_SIZE_EDGES
    edge_out = torch.empty(num_edges, dtype=dtype, device=in_3.device)

    gnn_edge_norm_kernel_10[(num_programs_edges,)](
        norm_ptr=norm, row_ptr=in_5, col_ptr=in_2, edge_weight_ptr=in_4, out_ptr=edge_out,
        num_edges=num_edges,
        BLOCK_SIZE=BLOCK_SIZE_EDGES,
    )

    return edge_out

@torch.fx.wrap
def gnn_norm_linear_10(in_0, in_1, in_2, in_3, in_4, in_5):
    edge_out = gnn_norm_fused_10(in_3, in_2, in_4, in_5)
    linear_out = torch.nn.functional.linear(in_1, in_0, None)
    return (edge_out, linear_out)

def replacement_func():
    return gnn_norm_linear_10