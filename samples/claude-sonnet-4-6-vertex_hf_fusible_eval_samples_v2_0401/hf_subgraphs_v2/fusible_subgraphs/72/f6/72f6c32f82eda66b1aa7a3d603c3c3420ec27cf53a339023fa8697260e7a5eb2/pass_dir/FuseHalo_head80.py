"""
Fused pass for eca_halonext26ts halo attention (float16, C=640, HEAD_SIZE=80).

Fuses: conv2d -> pad([2,2,2,2]) -> unfold(2,12,8) -> unfold(3,12,8)
        -> reshape(8,80,4,-1) -> permute(0,2,3,1) -> split([16,64],dim=-1)
        -> split[0].transpose(-1,-2)

Returns: (out1=[8,4,16,144], out2=[8,4,144,64])

Key insight:
  - unfold gives non-contiguous tensor → reshape must copy
  - We fuse pad + reshape-copy + permute + split + transpose into one Triton kernel
  - out1[head, win, q, pos]  = conv[0, head*80+q,  h_win*8+kh-2, w_win*8+kw-2]
  - out2[head, win, pos, kv] = conv[0, head*80+16+kv, h_win*8+kh-2, w_win*8+kw-2]
  - win = h_win*2 + w_win, pos = kh*12 + kw
  - Out-of-bounds → 0 (padding)
"""

import torch
import triton
import triton.language as tl


# ------------------------------------------------------------------ #
#  Pattern to match                                                    #
# ------------------------------------------------------------------ #
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


# ------------------------------------------------------------------ #
#  Triton kernel: fully fused 1×1 conv + halo post-processing         #
# ------------------------------------------------------------------ #
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_K': 64}, num_warps=8),
        triton.Config({'BLOCK_K': 128}, num_warps=8),
    ],
    key=['Cin'],
)
@triton.jit
def fused_conv_halo_kernel_80(
    w_ptr,      # [Cout, Cin]  weight (2-D view of the 1×1 conv weight)
    x_ptr,      # [Cin, H*W]   input  (squeezed batch dim, flattened HW)
    out1_ptr,   # [8, 4, 16, 144]
    out2_ptr,   # [8, 4, 144, 64]
    Cin: tl.constexpr,        # 512
    H: tl.constexpr,          # 16
    W: tl.constexpr,          # 16
    HEAD_SIZE: tl.constexpr,  # 80
    Q_SIZE: tl.constexpr,     # 16
    KV_SIZE: tl.constexpr,    # 64
    N_WINS: tl.constexpr,     # 4
    N_POS: tl.constexpr,      # 144
    WIN_S: tl.constexpr,      # 8
    PAD: tl.constexpr,        # 2
    KW: tl.constexpr,         # 12
    BLOCK_C: tl.constexpr,    # 16  (= Q_SIZE; evenly divides HEAD_SIZE)
    BLOCK_K: tl.constexpr,    # chosen by autotune
):
    """
    Grid: (Cout // BLOCK_C, N_WINS, N_POS)
    Each program computes BLOCK_C output channels for one (win, pos).

    Inner loop over Cin (BLOCK_K tiles) accumulates the dot product
    w[c_start:c_start+BLOCK_C, :] @ x[:, hw_idx] for the given spatial
    position, then scatters results to out1 (Q channels) or out2 (KV).
    Padding is handled implicitly: out-of-bounds → accumulator stays 0.
    """
    c_tile = tl.program_id(0)   # 0 .. Cout//BLOCK_C - 1
    win    = tl.program_id(1)   # 0 .. 3
    pos    = tl.program_id(2)   # 0 .. 143

    # ---- channel indices for this tile --------------------------------
    c_start  = c_tile * BLOCK_C
    c_offs   = c_start + tl.arange(0, BLOCK_C)   # [BLOCK_C]

    # ---- head / channel-within-head -----------------------------------
    head       = c_start // HEAD_SIZE             # which attention head
    ch_in_head = c_start - head * HEAD_SIZE       # 0 or Q_SIZE or 2*Q_SIZE …

    # ---- decode spatial position --------------------------------------
    h_win = win // 2
    w_win = win % 2
    kh    = pos // KW
    kw    = pos % KW

    h_src = h_win * WIN_S + kh - PAD   # may be < 0 or >= H (padding)
    w_src = w_win * WIN_S + kw - PAD

    in_bounds = (h_src >= 0) & (h_src < H) & (w_src >= 0) & (w_src < W)

    # Clamp to valid range; masked-zero the result afterwards
    safe_h  = tl.where(in_bounds, h_src, 0)
    safe_w  = tl.where(in_bounds, w_src, 0)
    hw_idx  = safe_h * W + safe_w          # column index into x

    # ---- GEMM inner loop (accumulate over Cin) -----------------------
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)

    for k_start in tl.range(0, Cin, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < Cin

        # x[k_offs, hw_idx] – one column of the input matrix
        x_vals = tl.load(x_ptr + k_offs * (H * W) + hw_idx,
                         mask=k_mask, other=0.0).to(tl.float32)

        # w[c_offs, k_offs] – a [BLOCK_C, BLOCK_K] weight tile
        w_offs = c_offs[:, None] * Cin + k_offs[None, :]
        w_vals = tl.load(w_ptr + w_offs,
                         mask=k_mask[None, :], other=0.0).to(tl.float32)

        # Partial dot products: acc[c] += sum_k w[c,k] * x[k]
        acc += tl.sum(w_vals * x_vals[None, :], axis=1)

    # Zero out padding positions
    acc = tl.where(in_bounds, acc, tl.zeros([BLOCK_C], dtype=tl.float32))

    # Cast to output dtype
    acc_out = acc.to(out1_ptr.dtype.element_ty)

    lane = tl.arange(0, BLOCK_C)

    # ---- is this tile Q (ch_in_head == 0) or KV (ch_in_head > 0)? ---
    is_q = ch_in_head < Q_SIZE

    # --- write Q channels to out1[head, win, ch_in_head+lane, pos] ---
    q_ch_safe = (ch_in_head + lane) % Q_SIZE      # keeps index in [0,Q_SIZE)
    out1_offs = (head * (N_WINS * Q_SIZE * N_POS)
                 + win  * (Q_SIZE * N_POS)
                 + q_ch_safe * N_POS
                 + pos)
    tl.store(out1_ptr + out1_offs, acc_out, mask=is_q)

    # --- write KV channels to out2[head, win, pos, kv_ch] ------------
    kv_ch     = ch_in_head - Q_SIZE + lane        # < 0 when is_q (never stored)
    kv_ch_safe = tl.where(kv_ch >= 0, kv_ch, 0)  # clamp negative for safety
    out2_offs = (head * (N_WINS * N_POS * KV_SIZE)
                 + win  * (N_POS * KV_SIZE)
                 + pos  * KV_SIZE
                 + kv_ch_safe)
    tl.store(out2_ptr + out2_offs, acc_out, mask=(is_q == 0))


# ------------------------------------------------------------------ #
#  Replacement kernel wrapper                                          #
# ------------------------------------------------------------------ #
@torch.fx.wrap
def fused_halo_80(in_0, in_1):
    """
    in_0: weight  [640, 512, 1, 1]
    in_1: input   [1, 512, 16, 16]
    """
    Cout = in_0.shape[0]   # 640
    Cin  = in_0.shape[1]   # 512
    H    = in_1.shape[2]   # 16
    W    = in_1.shape[3]   # 16

    dev   = in_1.device
    dtype = in_1.dtype

    # Prepare weight: [Cout, Cin, 1, 1] → [Cout, Cin] on the same device
    w = in_0.reshape(Cout, Cin).to(device=dev, dtype=dtype)
    # Prepare input: [1, Cin, H, W] → [Cin, H*W]
    x = in_1.reshape(Cin, H * W)

    NUM_HEADS = 8
    HEAD_SIZE = 80    # 640 // 8
    Q_SIZE    = 16
    KV_SIZE   = 64
    N_WINS    = 4
    N_POS     = 144
    BLOCK_C   = 16    # = Q_SIZE; evenly divides HEAD_SIZE (80/16=5)

    out1 = torch.empty((NUM_HEADS, N_WINS, Q_SIZE, N_POS), dtype=dtype, device=dev)
    out2 = torch.empty((NUM_HEADS, N_WINS, N_POS, KV_SIZE), dtype=dtype, device=dev)

    C_TILES = Cout // BLOCK_C   # 640 // 16 = 40
    grid = (C_TILES, N_WINS, N_POS)   # (40, 4, 144) = 23 040 programs

    fused_conv_halo_kernel_80[grid](
        w, x, out1, out2,
        Cin=Cin, H=H, W=W,
        HEAD_SIZE=HEAD_SIZE, Q_SIZE=Q_SIZE, KV_SIZE=KV_SIZE,
        N_WINS=N_WINS, N_POS=N_POS,
        WIN_S=8, PAD=2, KW=12,
        BLOCK_C=BLOCK_C,
    )

    return out1, out2


def replacement_func():
    return fused_halo_80