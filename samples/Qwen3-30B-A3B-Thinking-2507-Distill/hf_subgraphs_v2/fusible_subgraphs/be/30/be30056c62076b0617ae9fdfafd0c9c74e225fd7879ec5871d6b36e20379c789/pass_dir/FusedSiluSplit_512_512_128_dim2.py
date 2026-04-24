import torch
import triton
import triton.language as tl


# ─── Pattern ───────────────────────────────────────────────────────────────────
def pattern(in_1):
    tmp_1 = torch.nn.functional.silu(in_1, inplace=True)
    split = torch.functional.split(tmp_1, [512, 512, 128], dim=2)
    tmp_3 = split[0]
    tmp_4 = split[1]
    tmp_5 = split[2]
    tmp_6 = tmp_5.unsqueeze(2)
    return (tmp_3, tmp_4, tmp_6)


def replacement_args(in_1):
    return (in_1,)


# ─── Triton kernel ─────────────────────────────────────────────────────────────
# One program per (batch, row) pair.
# Each program reads one row of in_1 (1152 elements = 512 + 512 + 128),
# applies SiLU, and writes to the three output tensors.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
    ],
    key=['B'],
)
@triton.jit
def fused_silu_split_kernel(
    in_ptr,
    out0_ptr,   # [B, H, 512]
    out1_ptr,   # [B, H, 512]
    out2_ptr,   # [B, H, 128]  (reshaped to [B,H,1,128] in wrapper)
    B,
    H,
    W,
    HW,
    CHW2,
    C0: tl.constexpr,   # 512
    C1: tl.constexpr,   # 512
    C2: tl.constexpr,   # 128
    BLOCK_SIZE: tl.constexpr,
):
    bh = tl.program_id(0)
    b  = bh // H
    h  = bh  % H

    row_start = bh * W   # flat start in [B, H, W]

    # ── full row range (offsets 0..CHW2-1 within the row) ──────────────────
    offsets   = tl.arange(0, BLOCK_SIZE)
    row_offs  = offsets + row_start
    mask      = offsets < CHW2

    # ── load ─────────────────────────────────────────────────────────────────
    x = tl.load(in_ptr + row_offs, mask=mask, other=0.0)

    # ── SiLU: x * sigmoid(x) ─────────────────────────────────────────────────
    silu_x = x * tl.sigmoid(x)

    # ── segment 0 : channels 0..C0-1  → out0 [B,H,C0] ───────────────────────
    out0_idx = bh * C0 + offsets
    tl.store(out0_ptr + out0_idx, silu_x, mask=mask & (offsets < C0))

    # ── segment 1 : channels C0..C0+C1-1  → out1 [B,H,C1] ───────────────────
    out1_idx = bh * C1 + (offsets - C0)
    tl.store(out1_ptr + out1_idx, silu_x, mask=mask & (offsets >= C0) & (offsets < C0 + C1))

    # ── segment 2 : channels C0+C1..CHW2-1  → out2 [B,H,C2] ─────────────────
    C01 = C0 + C1
    out2_idx = bh * C2 + (offsets - C01)
    tl.store(out2_ptr + out2_idx, silu_x, mask=mask & (offsets >= C01))


# ─── Kernel wrapper ────────────────────────────────────────────────────────────
@torch.fx.wrap
def fused_silu_split(in_1):
    B   = in_1.shape[0]
    H   = in_1.shape[1]
    W   = in_1.shape[2]
    HW  = H * W
    CHW2 = HW                          # = 1152
    C0, C1, C2 = 512, 512, 128

    out0 = torch.empty(B, H, C0, dtype=in_1.dtype, device=in_1.device)
    out1 = torch.empty(B, H, C1, dtype=in_1.dtype, device=in_1.device)
    out2 = torch.empty(B, H, C2, dtype=in_1.dtype, device=in_1.device)

    grid = (B * H,)

    fused_silu_split_kernel[grid](
        in_1,
        out0, out1, out2,
        B, H, W, HW, CHW2,
        C0=C0, C1=C1, C2=C2,
    )

    # unsqueeze dim 2 for the 128-channel output → [B, H, 1, 128]
    tmp_6 = out2.reshape(B, H, 1, C2)

    return (out0, out1, tmp_6)


# ─── Replacement entry point ───────────────────────────────────────────────────
def replacement_func():
    return fused_silu_split