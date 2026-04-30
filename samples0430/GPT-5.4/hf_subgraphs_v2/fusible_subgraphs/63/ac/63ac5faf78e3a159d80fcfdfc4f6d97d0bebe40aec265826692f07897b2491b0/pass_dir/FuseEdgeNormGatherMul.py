import torch
import triton
import triton.language as tl


def pattern(tmp_2, in_2, in_4, in_5):
    tmp_5 = tmp_2[in_5]
    tmp_6 = tmp_5 * in_4
    tmp_7 = tmp_2[in_2]
    tmp_8 = tmp_6 * tmp_7
    return tmp_8


def replacement_args(tmp_2, in_2, in_4, in_5):
    return (tmp_2, in_2, in_4, in_5)


@triton.jit
def fused_edge_norm_kernel(
    norm_ptr,
    col_ptr,
    edge_weight_ptr,
    row_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    row_idx = tl.load(row_ptr + offs, mask=mask, other=0).to(tl.int32)
    col_idx = tl.load(col_ptr + offs, mask=mask, other=0).to(tl.int32)

    norm_row = tl.load(norm_ptr + row_idx, mask=mask, other=0).to(tl.float32)
    norm_col = tl.load(norm_ptr + col_idx, mask=mask, other=0).to(tl.float32)
    edge_weight = tl.load(edge_weight_ptr + offs, mask=mask, other=0).to(tl.float32)

    out = edge_weight * norm_row * norm_col

    tl.store(out_ptr + offs, out, mask=mask)


@torch.fx.wrap
def fused_edge_norm(norm, col, edge_weight, row):
    n_elements = edge_weight.numel()
    out = torch.empty_like(edge_weight)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    fused_edge_norm_kernel[grid](
        norm_ptr=norm,
        col_ptr=col,
        edge_weight_ptr=edge_weight,
        row_ptr=row,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=256,
        num_warps=4,
    )
    return out


def replacement_func():
    return fused_edge_norm