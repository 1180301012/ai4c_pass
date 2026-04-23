import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    in_1 += in_0
    in_2 = in_1
    tmp_1 = in_2.float()
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = tmp_2.type_as(in_2)
    tmp_4 = torch.nn.functional.dropout(tmp_3, p=0.1, training=False)
    return (tmp_4,)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_add_softmax_kernel(
    in0_ptr,
    in1_ptr,
    out_ptr,
    num_rows,
    row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    if row_id >= num_rows:
        return

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    base = row_id * row_stride + cols

    x0 = tl.load(in0_ptr + base, mask=mask, other=0.0)
    x1 = tl.load(in1_ptr + base, mask=mask, other=0.0)

    summed = x1 + x0
    vals = summed.to(tl.float32)
    vals = tl.where(mask, vals, float('-inf'))

    row_max = tl.max(vals, axis=0)
    shifted = vals - row_max
    exp_vals = tl.exp(shifted)
    denom = tl.sum(exp_vals, axis=0)
    probs = exp_vals / denom

    tl.store(out_ptr + base, probs, mask=mask)


@torch.fx.wrap
def fused_add_softmax(in_0, in_1):
    last_dim = in_0.shape[-1]
    num_rows = in_0.numel() // last_dim
    out = torch.empty_like(in_1)

    if last_dim <= 8:
        block_size = 8
        num_warps = 1
    elif last_dim <= 16:
        block_size = 16
        num_warps = 1
    elif last_dim <= 32:
        block_size = 32
        num_warps = 1
    else:
        block_size = 64
        num_warps = 2

    grid = (num_rows,)
    fused_add_softmax_kernel[grid](
        in_0,
        in_1,
        out,
        num_rows,
        last_dim,
        last_dim,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return (out,)


def replacement_func():
    return fused_add_softmax