"""
Shared Triton kernel and dispatch wrapper for Halo Attention KV projection.
Imported by HaloConvKV_C640_f16.py and HaloConvKV_C384.py so that
both passes share the SAME replacement_func() object (same Python identity),
staying within the replacement_func_limit.
"""

import torch
import triton
import triton.language as tl


# -----------------------------------------------------------------------
# Triton kernel: rearrange padded NCHW tensor -> K / V output layout
#
# Reads:  x [1, C_out, 20, 20]  (padded conv2d output)
# Writes: K [8, 4, K_DIM, 144]
#         V [8, 4, 144,  V_DIM]
#
# Grid: (8 heads, 4 windows, 9 pos-tiles)  → 288 blocks
# Each pos-tile covers BLOCK_POS=16 positions  (9 × 16 = 144 total)
# -----------------------------------------------------------------------

@triton.jit
def _halo_gather_kernel(
    x_ptr,
    K_ptr,
    V_ptr,
    HEAD_DIM:  tl.constexpr,  # C_out // 8
    K_DIM:     tl.constexpr,  # 16
    V_DIM:     tl.constexpr,  # 32 or 64
    PAD_HW:    tl.constexpr,  # 400  (20 × 20)
    PAD_W:     tl.constexpr,  # 20
    WIN_STEP:  tl.constexpr,  # 8
    WIN_SIZE:  tl.constexpr,  # 12
    BLOCK_POS: tl.constexpr,  # 16
):
    b   = tl.program_id(0)   # head   [0, 8)
    win = tl.program_id(1)   # window [0, 4)
    pt  = tl.program_id(2)   # pos-tile [0, 9)

    lp = pt * BLOCK_POS + tl.arange(0, BLOCK_POS)  # [16], local position

    ph = lp // WIN_SIZE   # row within 12×12 window
    pw = lp % WIN_SIZE    # col within 12×12 window

    wh = win // 2   # window row  {0, 1}
    ww = win % 2    # window col  {0, 1}

    # Padded-space coordinates — always valid (no boundary check needed)
    h_pad   = wh * WIN_STEP + ph       # [0, 20)
    w_pad   = ww * WIN_STEP + pw       # [0, 20)
    spatial = h_pad * PAD_W + w_pad    # [BLOCK_POS] linear index in H×W

    k_range = tl.arange(0, K_DIM)     # [K_DIM = 16]
    v_range = tl.arange(0, V_DIM)     # [V_DIM = 32 or 64]

    # ---- Load + store K ------------------------------------------------
    # x[0, b*HEAD_DIM + k, h_pad, w_pad] = x_ptr + (b*HEAD_DIM+k)*PAD_HW + spatial
    x_K = x_ptr + b * HEAD_DIM * PAD_HW
    K_data = tl.load(x_K + k_range[:, None] * PAD_HW + spatial[None, :])
    # K_data: [K_DIM, BLOCK_POS]

    K_base = K_ptr + b * (4 * K_DIM * 144) + win * (K_DIM * 144)
    tl.store(K_base + k_range[:, None] * 144 + lp[None, :], K_data)

    # ---- Load + store V ------------------------------------------------
    x_V = x_ptr + (b * HEAD_DIM + K_DIM) * PAD_HW
    V_data = tl.load(x_V + v_range[:, None] * PAD_HW + spatial[None, :])
    # V_data: [V_DIM, BLOCK_POS]  → transpose before writing V[b,win,lp,v]

    V_base = V_ptr + b * (4 * 144 * V_DIM) + win * (144 * V_DIM)
    tl.store(
        V_base + lp[:, None] * V_DIM + v_range[None, :],
        tl.trans(V_data),             # [BLOCK_POS, V_DIM]
    )


# -----------------------------------------------------------------------
# Python wrappers for each configuration
# -----------------------------------------------------------------------

def _run_c640_f16(x):
    """x: [1, 640, 20, 20] float16"""
    device = x.device
    x = x.contiguous()
    K_out = torch.empty((8, 4,  16, 144), dtype=torch.float16, device=device)
    V_out = torch.empty((8, 4, 144,  64), dtype=torch.float16, device=device)
    _halo_gather_kernel[(8, 4, 9)](
        x, K_out, V_out,
        HEAD_DIM=80, K_DIM=16, V_DIM=64,
        PAD_HW=400, PAD_W=20, WIN_STEP=8, WIN_SIZE=12, BLOCK_POS=16,
    )
    return K_out, V_out


def _run_c384(x):
    """x: [1, 384, 20, 20]; dtype = bf16 or f32"""
    device = x.device
    dtype  = x.dtype
    x = x.contiguous()
    K_out = torch.empty((8, 4,  16, 144), dtype=dtype, device=device)
    V_out = torch.empty((8, 4, 144,  32), dtype=dtype, device=device)
    _halo_gather_kernel[(8, 4, 9)](
        x, K_out, V_out,
        HEAD_DIM=48, K_DIM=16, V_DIM=32,
        PAD_HW=400, PAD_W=20, WIN_STEP=8, WIN_SIZE=12, BLOCK_POS=16,
    )
    return K_out, V_out


# -----------------------------------------------------------------------
# Single shared dispatch wrapper
# Both pass files import THIS exact function object so the framework
# sees only ONE replacement_func across all passes.
# -----------------------------------------------------------------------

@torch.fx.wrap
def halo_dispatch(x, route):
    if route == "c640_f16":
        return _run_c640_f16(x)
    elif route == "c384":
        return _run_c384(x)
    # Fallback (unreachable in practice)
    return _run_c384(x)