import torch
import triton
import triton.language as tl


def pattern(tmp_2, in_4, in_5, in_2):
    tmp_5 = tmp_2[in_5]
    tmp_6 = tmp_5 * in_4
    tmp_7 = tmp_2[in_2]
    tmp_8 = tmp_6 * tmp_7
    return tmp_8


def replacement_args(tmp_2, in_4, in_5, in_2):
    return (tmp_2, in_5, in_2, in_4)


@triton.jit
def gnn_edge_norm_kernel(
    norm_ptr, row_ptr, col_ptr, ew_ptr, out_ptr,
    num_edges,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_edges

    row_idx = tl.load(row_ptr + offsets, mask=mask, other=0)
    col_idx = tl.load(col_ptr + offsets, mask=mask, other=0)
    ew = tl.load(ew_ptr + offsets, mask=mask, other=0.0)

    norm_row = tl.load(norm_ptr + row_idx)
    norm_col = tl.load(norm_ptr + col_idx)

    result = norm_row * ew * norm_col

    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def gnn_edge_norm(norm, row, col, edge_weight):
    norm = norm.contiguous()
    row = row.contiguous()
    col = col.contiguous()
    edge_weight = edge_weight.contiguous()

    num_edges = row.shape[0]
    BLOCK_SIZE = 256
    num_programs = (num_edges + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(edge_weight)

    if num_programs > 0:
        gnn_edge_norm_kernel[(num_programs,)](
            norm_ptr=norm, row_ptr=row, col_ptr=col, ew_ptr=edge_weight, out_ptr=out,
            num_edges=num_edges,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return out


def replacement_func():
    return gnn_edge_norm