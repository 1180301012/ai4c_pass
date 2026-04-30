import torch
import triton
import triton.language as tl


def pattern(x):
    tmp_3 = x[(slice(None, None, None), slice(1, None, None))]
    tmp_4 = torch._C._nn.pad(tmp_3, [0, 0, 0, 1, 0, 0], 'constant', 0.0)
    tmp_5 = x[(slice(None, None, None), slice(None, -1, None))]
    tmp_6 = torch._C._nn.pad(tmp_5, [0, 0, 1, 0, 0, 0], 'constant', 0.0)
    tmp_7 = torch.cat([tmp_4, x, tmp_6], dim=2)
    return tmp_7


def replacement_args(x):
    return (x,)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_S": 1, "BLOCK_D": 128}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_S": 2, "BLOCK_D": 128}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_S": 4, "BLOCK_D": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_S": 8, "BLOCK_D": 128}, num_warps=4, num_stages=2),
    ],
    key=["S", "D"],
)
@triton.jit
def _shift_pad_cat_kernel(
    x_ptr,
    out_ptr,
    B,
    S,
    D,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    num_s_blocks = tl.cdiv(S, BLOCK_S)
    b = pid // num_s_blocks
    sb = pid % num_s_blocks
    s0 = sb * BLOCK_S

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D

    def load_row(s):
        valid_s = (s >= 0) & (s < S)
        safe_s = tl.where(valid_s, s, 0)
        row_base = (b * S + safe_s) * D
        return tl.load(x_ptr + row_base + offs_d, mask=d_mask & valid_s, other=0.0)

    row_prev = load_row(s0 - 1)
    row_center = load_row(s0)
    row_next = load_row(s0 + 1)

    for i in range(BLOCK_S):
        s = s0 + i
        valid_out = s < S
        out_base = (b * S + s) * (3 * D)
        tl.store(out_ptr + out_base + offs_d, row_next, mask=d_mask & valid_out)
        tl.store(out_ptr + out_base + D + offs_d, row_center, mask=d_mask & valid_out)
        tl.store(out_ptr + out_base + 2 * D + offs_d, row_prev, mask=d_mask & valid_out)
        row_prev = row_center
        row_center = row_next
        row_next = load_row(s0 + i + 2)


@torch.fx.wrap
def fused_shift_pad_cat_seqdim1(x):
    B, S, D = x.shape
    out = torch.empty((B, S, 3 * D), device=x.device, dtype=x.dtype)

    grid = lambda meta: (B * triton.cdiv(S, meta["BLOCK_S"]),)
    _shift_pad_cat_kernel[grid](
        x,
        out,
        B,
        S,
        D,
    )
    return out


def replacement_func():
    return fused_shift_pad_cat_seqdim1