import torch
import triton
import triton.language as tl


# ── Pattern for float16 graph: reshape(8,80,4,-1), split([16,64]) ──────────

def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.nn.functional.pad(conv2d, [2, 2, 2, 2], 'constant', None)
    tmp_3 = tmp_2.unfold(2, 12, 8)
    tmp_4 = tmp_3.unfold(3, 12, 8)
    tmp_5 = tmp_4.reshape(8, 80, 4, -1)
    tmp_6 = tmp_5.permute(0, 2, 3, 1)
    split = torch.functional.split(tmp_6, [16, 64], dim=-1)
    tmp_8 = split[0]
    tmp_9 = split[1]
    tmp_10 = tmp_8.transpose(-1, -2)
    return (tmp_10, tmp_9)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ── Triton kernel ────────────────────────────────────────────────────────────
# Each pid_b ∈ [0, 16) corresponds to one (wh, ww) unfold window.
# For that window we iterate over the 12×12 kernel positions (with boundary check)
# and accumulate a GEMM over C_in channels.
#
# Output mapping (for HW=256, M=49):
#   out1[b, wh, ww, c]   = acc[wh*49+ww, c]           shape [8, 49, 4, 16]
#   out2[b, wh, ww, c]   = acc[wh*49+ww, 16+c]        shape [8, 4, 49, 64]
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C_IN': 32,  'BLOCK_C_OUT': 32, 'BLOCK_M': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_C_IN': 64,  'BLOCK_C_OUT': 32, 'BLOCK_M': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_C_IN': 32,  'BLOCK_C_OUT': 64, 'BLOCK_M': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_C_IN': 64,  'BLOCK_C_OUT': 64, 'BLOCK_M': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_C_IN': 128, 'BLOCK_C_OUT': 32, 'BLOCK_M': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_C_IN': 32,  'BLOCK_C_OUT': 32, 'BLOCK_M': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_C_IN': 128, 'BLOCK_C_OUT': 64, 'BLOCK_M': 16}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_C_IN': 64,  'BLOCK_C_OUT': 64, 'BLOCK_M': 16}, num_stages=3, num_warps=8),
    ],
    key=['C_in', 'C_out', 'HW'],
)
@triton.jit
def _fused_conv_unfold_f16(
    in1_ptr,   # [1, C_in, H, W]
    in0_ptr,   # [C_out, C_in, 1, 1]
    out1_ptr,  # [8, N_W, 4, 16]   (tmp_10 after transpose(-1,-2))
    out2_ptr,  # [8, 4, N_W, 64]  (tmp_9)
    C_in, C_out, HW, W_in,
    N_W,       # = N_h = N_w = 4
    C_left,    # = 16
    C_right,   # = 64
    BLOCK_C_IN:  tl.constexpr,
    BLOCK_C_OUT: tl.constexpr,
    BLOCK_M:     tl.constexpr,
):
    # Grid: (ceil(HW/BLOCK_M),  ceil(C_out/BLOCK_C_OUT),  16 windows)
    pid_m    = tl.program_id(0)
    pid_c    = tl.program_id(1)
    pid_b    = tl.program_id(2)   # 0..15 → wh = pid_b // 4,  ww = pid_b % 4

    wh = pid_b // 4
    ww = pid_b % 4

    m_start  = pid_m  * BLOCK_M
    c_start  = pid_c  * BLOCK_C_OUT

    m_range  = m_start + tl.arange(0, BLOCK_M)
    c_range  = c_start  + tl.arange(0, BLOCK_C_OUT)

    m_mask   = m_range  < N_W * N_W   # valid unfold-window positions
    c_mask   = c_range  < C_out

    # Window offset in the padded input (padded by 2 on each side)
    win_h = wh * 8 + tl.arange(0, 12)[:, None] + 2   # [12, 1]
    win_w = ww * 8 + tl.arange(0, 12)[None, :] + 2   # [1, 12]
    h_in  = win_h - 2                                  # [12, 12] in [-2..17]
    w_in  = win_w - 2                                  # [12, 12] in [-2..17]
    valid = (h_in >= 0) & (h_in < N_W * 4 + 4) & (w_in >= 0) & (w_in < N_W * 4 + 4)
    hw_in = tl.where(valid, h_in * W_in + w_in, 0)   # [12, 12] flat input index (or 0 if OOB)

    acc = tl.zeros((BLOCK_M, BLOCK_C_OUT), dtype=tl.float32)

    for k in range(0, tl.cdiv(C_in, BLOCK_C_IN)):
        cin_range = k * BLOCK_C_IN + tl.arange(0, BLOCK_C_IN)
        cin_mask  = cin_range < C_in

        # Input tile [BLOCK_M, BLOCK_C_IN]: in_1[0, cin, h_in, w_in]
        a_offs = cin_range[None, :].to(tl.int64) * HW + hw_in[:, None].to(tl.int64)
        a_mask = m_mask[:, None] & cin_mask[None, :]
        a = tl.load(in1_ptr + a_offs, mask=a_mask, other=0.0)

        # Weight tile [BLOCK_C_OUT, BLOCK_C_IN]: in_0[c_out, cin, 0, 0]
        b_offs = c_range[:, None] * C_in + cin_range[None, :]
        b_mask = c_mask[:, None] & cin_mask[None, :]
        b = tl.load(in0_ptr + b_offs, mask=b_mask, other=0.0)

        acc = acc + tl.dot(a, tl.trans(b))

    # ── out1 [8, N_W, 4, 16]: out1[b, wh, ww, c] = acc[win+m, c] ────────────
    win_idx  = wh * N_W + ww                             # scalar
    out1_off = pid_b * (N_W * 4 * 16) + win_idx * (4 * 16) + m_range[:, None] * (4 * 16) + c_range[None, :]
    out1_mask = m_mask[:, None] & c_mask[None, :]
    tl.store(out1_ptr + out1_off, acc.to(tl.float16), mask=out1_mask)

    # ── out2 [8, 4, N_W, 64]: out2[b, wh, ww, c] = acc[win+m, C_left+c] ──────
    out2_off = pid_b * (4 * N_W * 64) + win_idx * (N_W * 64) + m_range[:, None] * (N_W * 64) + (c_range[None, :] - C_left)
    out2_mask = m_mask[:, None] & c_mask[None, :] & (c_range[None, :] >= C_left)
    tl.store(out2_ptr + out2_off, acc.to(tl.float16), mask=out2_mask)


@torch.fx.wrap
def fused_conv_unfold_f16(in_0, in_1):
    """
    in_0 : weight  [C_out, C_in, 1, 1]
    in_1 : input   [1, C_in, H, W]
    returns (tmp_10, tmp_9) matching the original float16 graph outputs
    """
    C_out = in_0.shape[0]
    C_in  = in_0.shape[1]
    H_in  = in_1.shape[2]
    W_in  = in_1.shape[3]
    HW    = H_in * W_in

    N_W    = 4          # N_h = N_w from unfold(2,12,8) / unfold(3,12,8)
    M_tot  = N_W * N_W  # 16 total unfold windows (= pid_b range)
    C_left = 16
    C_right= 64

    out1 = torch.empty((8, N_W, 4, C_left),    dtype=in_1.dtype, device=in_1.device)
    out2 = torch.empty((8, N_W, N_W, C_right), dtype=in_1.dtype, device=in_1.device)

    grid = lambda meta: (
        triton.cdiv(M_tot, meta['BLOCK_M']),
        triton.cdiv(C_out, meta['BLOCK_C_OUT']),
        16,
    )

    _fused_conv_unfold_f16[grid](
        in_1, in_0, out1, out2,
        C_in, C_out, HW, W_in,
        N_W, C_left, C_right,
    )

    return out1, out2


def replacement_func():
    return fused_conv_unfold_f16