import torch
import triton
import triton.language as tl


def pattern(in_1, in_2):
    """
    Match the fused pattern: view(-1, 1) -> mul
    """
    tmp_0 = in_1.view(-1, 1)
    tmp_1 = tmp_0 * in_2
    return tmp_1


def replacement_args(in_1, in_2):
    return (in_1, in_2)


@triton.jit
def mul_broadcast_kernel(
    weight_ptr,
    feat_ptr,
    out_ptr,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)
    offset = row_idx * n_cols + col_idx
    weight = tl.load(weight_ptr + row_idx)
    feat = tl.load(feat_ptr + offset)
    result = weight * feat
    tl.store(out_ptr + offset, result)


def mul_broadcast_wrapper(in_1, in_2):
    n_rows = in_1.shape[0]
    n_cols = in_2.shape[1]
    out = torch.empty((n_rows, n_cols), dtype=in_2.dtype, device=in_2.device)
    grid = (n_rows, n_cols)
    mul_broadcast_kernel[grid](
        weight_ptr=in_1,
        feat_ptr=in_2,
        out_ptr=out,
        n_rows=n_rows,
        n_cols=n_cols,
    )
    return out


def replacement_func():
    return mul_broadcast_wrapper