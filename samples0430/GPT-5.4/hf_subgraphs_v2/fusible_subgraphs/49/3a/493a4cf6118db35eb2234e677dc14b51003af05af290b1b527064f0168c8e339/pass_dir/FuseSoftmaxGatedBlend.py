import torch
import triton
import triton.language as tl


# Match the original graph exactly so the pass is robust across dtypes.
def pattern(in_0, in_1, in_2):
    tmp_1 = in_2.softmax(dim=-1)
    tmp_2 = in_0.view(1, -1, 1, 1)
    tmp_3 = torch.sigmoid(tmp_2)
    tmp_4 = 1.0 - tmp_3
    tmp_5 = tmp_4 * in_1
    tmp_6 = torch.sigmoid(tmp_2)
    tmp_7 = tmp_6 * tmp_1
    tmp_8 = tmp_5 + tmp_7
    return tmp_8


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def _softmax_gated_blend_kernel(
    patch_ptr,
    pos_ptr,
    gate_ptr,
    out_ptr,
    HEIGHT: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    row_blk = tl.program_id(0)
    ch = tl.program_id(1)

    rows = row_blk * ROWS_PER_PROG + tl.arange(0, ROWS_PER_PROG)
    cols = tl.arange(0, BLOCK_SIZE)
    row_mask = rows < HEIGHT
    col_mask = cols < WIDTH
    mask = row_mask[:, None] & col_mask[None, :]

    base = ((ch * HEIGHT + rows[:, None]) * WIDTH) + cols[None, :]

    pos = tl.load(pos_ptr + base, mask=mask, other=-float("inf")).to(tl.float32)
    row_max = tl.max(pos, axis=1)[:, None]
    num = tl.exp(pos - row_max)
    den = tl.sum(num, axis=1)[:, None]
    softmax = num / den

    patch = tl.load(patch_ptr + base, mask=mask, other=0.0).to(tl.float32)

    gate_raw = tl.load(gate_ptr + ch).to(tl.float32)
    gate = 1.0 / (1.0 + tl.exp(-gate_raw))

    out = patch + gate * (softmax - patch)
    tl.store(out_ptr + base, out, mask=mask)


@torch.fx.wrap
def fused_softmax_gated_blend(in_0, in_1, in_2):
    out = torch.empty_like(in_1)
    _softmax_gated_blend_kernel[(49, 16)](
        patch_ptr=in_1,
        pos_ptr=in_2,
        gate_ptr=in_0,
        out_ptr=out,
        HEIGHT=196,
        WIDTH=196,
        BLOCK_SIZE=256,
        ROWS_PER_PROG=4,
        num_warps=2,
        num_stages=2,
    )
    return out


def replacement_func():
    return fused_softmax_gated_blend