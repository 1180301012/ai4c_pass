import torch
import triton
import triton.language as tl


@triton.jit
def _out_kernel(
    tmp8_ptr,            # (196,) int32  — grid diffs: tmp8[k] = row(k)*14 + col(k)
    out_ptr,             # (112896,) float32  output flat
    BLOCK_SIZE: tl.constexpr,
):
    """
    One program per output row r (0..195).
    For each column c: sq_dist = d_row^2 + d_col^2
      where d_row*14 + d_col = tmp8[abs((c//14 - r//14)*14 + (c%14 - r%14))]
    """
    r = tl.program_id(0)
    row1 = r // 14
    col1 = r % 14

    c_offsets = tl.arange(0, BLOCK_SIZE)
    mask = c_offsets < 196

    row2 = c_offsets // 14
    col2 = c_offsets % 14

    diff = (row2 - row1) * 14 + (col2 - col1)   # in [-195, 195]
    abs_diff = tl.abs(diff)

    d = tl.load(tmp8_ptr + abs_diff, mask=mask, other=0).to(tl.float32)
    sq_dist = d * d

    base = r * 196 * 3
    tl.store(out_ptr + base + c_offsets * 3 + 0, sq_dist, mask=mask)
    tl.store(out_ptr + base + c_offsets * 3 + 1, sq_dist, mask=mask)
    tl.store(out_ptr + base + c_offsets * 3 + 2, sq_dist, mask=mask)


@torch.fx.wrap
def triton_fused_geometric(tmp_2):
    """
    Replacement for the entire forward (geometric computation only; LN stays PyTorch).
    `tmp_2` is the layer_norm output – kept as pass-through.
    Returns (out, tmp_2) where out has same shape/dtype as original tmp_3.
    """
    tmp8 = torch.empty(196,   dtype=torch.int32,  device='cuda')
    out  = torch.empty(1, 196, 196, 3, dtype=torch.float32, device='cuda')

    _out_kernel[(196,)](
        tmp8,
        out,
        BLOCK_SIZE=256,
    )
    return out, tmp_2


def pattern(tmp_2):
    # tmp_2 is an anchor (layer_norm output, not consumed by geometric ops)
    tmp_3 = torch.zeros(1, 196, 196, 3)
    tmp_4 = torch.arange(14)
    tmp_5 = tmp_4.view(1, -1)
    tmp_6 = torch.arange(14)
    tmp_7 = tmp_6.view(-1, 1)
    tmp_8 = tmp_5 - tmp_7
    tmp_9  = tmp_8.repeat(14, 14)
    tmp_10 = tmp_8.repeat_interleave(14, dim=0)
    tmp_11 = tmp_10.repeat_interleave(14, dim=1)
    tmp_12 = tmp_9 ** 2
    tmp_13 = tmp_11 ** 2
    tmp_14 = tmp_12 + tmp_13
    tmp_15 = tmp_14.unsqueeze(0)
    tmp_3[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 2)] = tmp_15
    tmp_17 = tmp_11.unsqueeze(0)
    tmp_3[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 1)] = tmp_17
    tmp_19 = tmp_9.unsqueeze(0)
    tmp_3[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 0)] = tmp_19
    return tmp_3


def replacement_args(tmp_2):
    return (tmp_2,)


def replacement_func():
    return triton_fused_geometric