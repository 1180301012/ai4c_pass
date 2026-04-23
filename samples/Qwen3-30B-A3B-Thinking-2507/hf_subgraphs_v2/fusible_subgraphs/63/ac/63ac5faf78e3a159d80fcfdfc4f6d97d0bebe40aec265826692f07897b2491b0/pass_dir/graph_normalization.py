import torch
from torch import inf
import triton
import triton.language as tl

def pattern(in_3, in_4, in_5, in_2):
    tmp_2 = in_3.pow_(-0.5)
    tmp_3 = tmp_2.eq(inf)
    tmp_4 = tmp_2.masked_fill_(tmp_3, 0)
    tmp_5 = tmp_4[in_5]
    tmp_6 = tmp_5 * in_4
    tmp_7 = tmp_4[in_2]
    tmp_8 = tmp_6 * tmp_7
    return tmp_8

def replacement_args(in_3, in_4, in_5, in_2):
    return (in_3, in_4, in_5, in_2)

@triton.jit
def compute_scaling_kernel(
    degree_ptr, 
    row_ptr,
    col_ptr,
    edge_weights_ptr,
    output_ptr,
    M,
    BLOCK_SIZE: tl.constexpr
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M

    row_indices = tl.load(row_ptr + offsets, mask=mask)
    col_indices = tl.load(col_ptr + offsets, mask=mask)
    weights = tl.load(edge_weights_ptr + offsets, mask=mask)

    d_row = tl.load(degree_ptr + row_indices, mask=mask)
    d_col = tl.load(degree_ptr + col_indices, mask=mask)

    scaling_row = tl.where(d_row == 0, 0.0, d_row ** -0.5)
    scaling_col = tl.where(d_col == 0, 0.0, d_col ** -0.5)

    scaling = scaling_row * scaling_col
    result = weights * scaling

    tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def graph_normalization_kernel(degree_vector, row_indices, col_indices, edge_weights):
    M = row_indices.numel()
    BLOCK_SIZE = 1024
    num_programs = (M + BLOCK_SIZE - 1) // BLOCK_SIZE

    output = torch.empty_like(edge_weights)
    compute_scaling_kernel[(num_programs,)](
        degree_vector, 
        row_indices, 
        col_indices, 
        edge_weights, 
        output,
        M,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output

def replacement_func():
    return graph_normalization_kernel