import torch
import triton
import triton.language as tl
from torch import device


# ---------------------------------------------------------------------------
# Pattern – mirrors model.py exactly (no None-cleanup lines)
# ---------------------------------------------------------------------------
def pattern(in_0, in_1):
    tmp_1 = in_1.cumsum(-1)
    tmp_2 = tmp_1 - 1
    tmp_3 = in_0.__eq__(0)
    tmp_4 = tmp_2.masked_fill_(tmp_3, 1)
    tmp_5 = tmp_4.unsqueeze(0)          # use tmp_4 (result of in-place op)
    tmp_6 = tmp_5.expand(3, -1, -1)
    tmp_7 = tmp_6.to(device(type='cuda', index=0))
    max_1 = tmp_7.max(0, keepdim=False)
    tmp_9 = max_1[0]
    max_2 = tmp_9.max(-1, keepdim=True)
    tmp_11 = max_2[0]
    tmp_12 = tmp_11 + 1
    tmp_13 = tmp_12 - 9
    return (tmp_13, tmp_7)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ---------------------------------------------------------------------------
# Triton kernel: fuses all ops in one pass over each row
#
#   • cumsum along last dim of in_1
#   • subtract 1
#   • masked_fill where in_0 == 0  →  set value to 1
#   • write result 3 times to produce contiguous [3, B, N] tensor (tmp_7)
#   • compute per-row max and store  tmp_13 = max - 8  (shape [B, 1])
# ---------------------------------------------------------------------------
@triton.jit
def _fused_kernel(
    in0_ptr,
    in1_ptr,
    out7_ptr,   # shape [3, B, N]
    out13_ptr,  # shape [B, 1]
    B,
    N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)          # one program per batch row
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    # ---- load ---------------------------------------------------------------
    in1_row = tl.load(in1_ptr + row * N + cols, mask=mask, other=0)
    in0_row = tl.load(in0_ptr + row * N + cols, mask=mask, other=1)

    # ---- cumsum(-1) - 1 -----------------------------------------------------
    cs   = tl.cumsum(in1_row, axis=0)
    tmp2 = cs - 1

    # ---- masked_fill: where in0 == 0, set to 1 ------------------------------
    one  = tl.full([BLOCK_N], 1, dtype=tl.int64)
    tmp2 = tl.where(in0_row == 0, one, tmp2)

    # ---- per-row max (only over valid positions) -----------------------------
    NEG_INF = tl.full([BLOCK_N], -4611686018427387904, dtype=tl.int64)
    valid   = tl.where(mask, tmp2, NEG_INF)
    row_max = tl.max(valid, axis=0)

    # ---- store tmp_13 = row_max + 1 - 9  (keepdim → shape [B,1]) -----------
    tl.store(out13_ptr + row, row_max - 8)

    # ---- store tmp_7: 3 identical slices [3, B, N] --------------------------
    base = row * N + cols
    tl.store(out7_ptr +           base, tmp2, mask=mask)
    tl.store(out7_ptr + B * N  + base, tmp2, mask=mask)
    tl.store(out7_ptr + 2*B*N + base, tmp2, mask=mask)


# ---------------------------------------------------------------------------
# Python wrapper (must be decorated with @torch.fx.wrap)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def _triton_fused_forward(in0, in1):
    B = in0.shape[0]
    N = in0.shape[1]

    # BLOCK_N must be a power-of-2 >= N for tl.arange and tl.cumsum
    BLOCK_N = max(triton.next_power_of_2(N), 16)
    num_warps = 4 if BLOCK_N <= 256 else 8

    out7  = torch.empty((3, B, N), dtype=torch.int64, device=in0.device)
    out13 = torch.empty((B, 1),    dtype=torch.int64, device=in0.device)

    _fused_kernel[(B,)](
        in0, in1, out7, out13,
        B, N,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )

    return out13, out7


# ---------------------------------------------------------------------------
# replacement_func – return the wrapper (do NOT call it)
# ---------------------------------------------------------------------------
def replacement_func():
    return _triton_fused_forward