import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_1 = in_0.ne(1)
    tmp_2 = tmp_1.int()
    tmp_3 = torch.cumsum(tmp_2, dim=1)
    tmp_4 = tmp_3.type_as(tmp_2)
    tmp_5 = tmp_4 + 0
    tmp_6 = tmp_5 * tmp_2
    tmp_7 = tmp_6.long()
    tmp_8 = tmp_7 + 1
    return (tmp_8,)


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def fused_cumsum_seq_kernel(
    in_ptr,
    out_ptr,
    num_rows,
    num_cols,
):
    pid = tl.program_id(0)
    if pid >= num_rows:
        return

    row_start = pid * num_cols
    cumsum = 0

    for col in range(num_cols):
        val = tl.load(in_ptr + row_start + col)
        mask_int = (val != 1).to(tl.int32)
        cumsum = cumsum + mask_int
        result = cumsum * mask_int + 1
        tl.store(out_ptr + row_start + col, result)


@torch.fx.wrap
def fused_ne_cumsum(in_0):
    num_rows = in_0.shape[0]
    num_cols = in_0.shape[1]
    out = torch.empty_like(in_0)

    grid = (num_rows,)

    fused_cumsum_seq_kernel[grid](
        in_ptr=in_0,
        out_ptr=out,
        num_rows=num_rows,
        num_cols=num_cols,
    )

    return (out,)


def replacement_func():
    return fused_ne_cumsum