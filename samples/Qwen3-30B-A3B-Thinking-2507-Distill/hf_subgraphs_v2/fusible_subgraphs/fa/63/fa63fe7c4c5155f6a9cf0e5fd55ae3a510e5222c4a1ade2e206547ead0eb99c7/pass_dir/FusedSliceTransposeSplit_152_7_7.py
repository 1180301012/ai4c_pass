import torch
import triton
import triton.language as tl


# ── Pattern: slice + transpose + reshape only (no split inside pattern) ──────
def pattern(in_2):
    tmp_2 = in_2[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)]
    tmp_3 = tmp_2.transpose(-1, -2)
    tmp_4 = tmp_3.reshape(1, 152, 7, 7)
    return tmp_4


def replacement_args(in_2):
    return (in_2, "152_7_7")


# ── Shared Triton kernel ──────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 64,  'BLOCK_D': 32}),
        triton.Config({'BLOCK_S': 128, 'BLOCK_D': 32}),
        triton.Config({'BLOCK_S': 256, 'BLOCK_D': 32}),
        triton.Config({'BLOCK_S': 32,  'BLOCK_D': 32}),
        triton.Config({'BLOCK_S': 512, 'BLOCK_D': 32}),
    ],
    key=['S1', 'BH'],
)
@triton.jit
def _v_transpose_kernel(
    in2_ptr,
    out_ptr,
    BH, S1, D, N,
    BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr,
):
    bh = tl.program_id(1)
    pid = tl.program_id(0)
    S_BLOCKS = tl.cdiv(S1, BLOCK_S)
    r_block = pid // S_BLOCKS
    s_block = pid % S_BLOCKS

    rows = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    cols = tl.arange(0, BLOCK_D)

    r_mask = rows < S1
    d_mask = cols < D
    mask = r_mask[:, None] & d_mask[None, :]

    in_row = r_block * BLOCK_S + rows
    # Use N for load stride (bh dimension): in_2 full stride = N*D (not S1*D)
    offsets = bh * (N * D) + in_row[:, None] * D + cols[None, :]
    data = tl.load(in2_ptr + offsets, mask=mask, other=0.0)

    # Store to transposed [BH, D, S1]: out[bh, d, r] = bh*D*S1 + d*S1 + r
    out_off = bh * (D * S1) + cols[None, :] * S1 + rows[:, None]  # [BLOCK_S, BLOCK_D]
    tl.store(out_ptr + out_off, data, mask=mask)


# ── Shared dispatch wrapper ───────────────────────────────────────────────────
@torch.fx.wrap
def fused_v_dispatch(in_2, route):
    B  = in_2.shape[0]   # 1
    H  = in_2.shape[1]   # BH
    N  = in_2.shape[2]   # full seq len (e.g. 50)
    D  = in_2.shape[3]   # head dim
    BH = B * H
    S1 = N - 1            # seq len after slice (e.g. 49)

    if route == "152_7_7":
        out = torch.empty((1, 152, 7, 7), dtype=in_2.dtype, device=in_2.device)
        H_out, W_out = 7, 7
    elif route == "320_7_7":
        out = torch.empty((1, 320, 7, 7), dtype=in_2.dtype, device=in_2.device)
        H_out, W_out = 7, 7
    elif route == "216_14_14":
        out = torch.empty((1, 216, 14, 14), dtype=in_2.dtype, device=in_2.device)
        H_out, W_out = 14, 14
    elif route == "256_14_14":
        out = torch.empty((1, 256, 14, 14), dtype=in_2.dtype, device=in_2.device)
        H_out, W_out = 14, 14
    else:
        out = torch.empty((1, 152, 7, 7), dtype=in_2.dtype, device=in_2.device)

    C_out = H_out * W_out
    grid = lambda meta: (BH * triton.cdiv(S1, meta['BLOCK_S']), BH)
    # Use N (full seq len) for load stride (data ptr already points to [:::, 1:, :])
    # Use S1 for store (output is [BH, D, S1])
    _v_transpose_kernel[grid](in_2, out, BH, S1, D, N)
    return out


def replacement_func():
    return fused_v_dispatch