import torch
import triton
import triton.language as tl


@triton.jit
def _masked_mean_kernel(
    in0_ptr,   # [B, S, H] int64 mask
    in1_ptr,   # [B, S, H] bfloat16/float16 hidden states
    out_ptr,   # [B, H] float32 output
    B, S, H,
    BLOCK_H: tl.constexpr,
):
    # grid = (B, H // BLOCK_H)
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    num = tl.zeros([BLOCK_H], dtype=tl.float32)
    den = tl.zeros([BLOCK_H], dtype=tl.float32)

    # Loop over exactly S sequence positions — no wasted padding loads.
    for s in range(S):
        base = (pid_b * S + s) * H + h_offs
        m = tl.load(in0_ptr + base, mask=h_mask, other=0).to(tl.float32)
        x = tl.load(in1_ptr + base, mask=h_mask, other=0.0).to(tl.float32)
        num += x * m
        den += m

    den = tl.maximum(den, 1e-9)
    tl.store(out_ptr + pid_b * H + h_offs, num / den, mask=h_mask)


@torch.fx.wrap
def _masked_mean_wrapper(in_0, in_1):
    B, S, H = in_0.shape
    out = torch.empty((B, H), dtype=torch.float32, device=in_0.device)

    # Choose the largest power-of-2 BLOCK_H that divides H and fits ≤ 1024.
    # For H=1024 → BLOCK_H=1024, 1 program per batch element (minimal launch overhead).
    BLOCK_H = 1
    while BLOCK_H * 2 <= min(H, 1024) and H % (BLOCK_H * 2) == 0:
        BLOCK_H *= 2

    # num_warps: 8 warps (256 threads) for BLOCK_H=1024 gives 4 elements/thread,
    # good balance of ILP and register pressure.
    nw = min(BLOCK_H // 32, 8) if BLOCK_H >= 32 else 1

    _masked_mean_kernel[(B, H // BLOCK_H)](
        in_0, in_1, out,
        B, S, H,
        BLOCK_H=BLOCK_H,
        num_warps=nw,
    )
    return out


def pattern(in_0, in_1):
    tmp_0 = in_0.to(torch.float32)
    tmp_1 = in_1 * tmp_0
    tmp_2 = torch.sum(tmp_1, 1)
    tmp_3 = tmp_0.sum(1)
    tmp_4 = torch.clamp(tmp_3, min=1e-09)
    tmp_5 = tmp_2 / tmp_4
    tmp_6 = torch.cat([tmp_5], 1)
    return tmp_6


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return _masked_mean_wrapper