import torch
import triton
import triton.language as tl


# ── Triton kernel ─────────────────────────────────────────────────────────────
# Fuses: 1x1-conv  +  pad[2,2,2,2]  +  unfold(2,12,8)  +  unfold(3,12,8)
#       +  reshape(8,48,4,-1)  +  permute(0,2,3,1)  +  split([16,32],-1)
#       +  transpose(-1,-2)  →  (out0=[8,4,16,4],  out1=[8,4,4,32])
#
# Input : in_1 [1, K, H_IN, W_IN],  in_0 [C_out, K, 1, 1]
# H_IN=W_IN=16,  C_out=48,  split0=16,  split1=32
# Padded size = 20x20.  Unfold patches = 2x2 (each 12x12, stride=8).
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_K': 32},  num_stages=2, num_warps=4),
        triton.Config({'BLOCK_K': 64},  num_stages=2, num_warps=4),
        triton.Config({'BLOCK_K': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_K': 32},  num_stages=3, num_warps=4),
        triton.Config({'BLOCK_K': 64},  num_stages=3, num_warps=4),
        triton.Config({'BLOCK_K': 32},  num_stages=2, num_warps=8),
        triton.Config({'BLOCK_K': 64},  num_stages=2, num_warps=8),
    ],
    key=['K'],
)
@triton.jit
def _fused_conv_patch_kernel_48_32(
    x_ptr, w_ptr, out_ptr0, out_ptr1,
    K,
    BLOCK_K: tl.constexpr,
):
    """
    Each program handles one (patch, spatial-block).
    BLOCK_HW = 16 (spatial positions per program), fixed.
    Processes C_out = 48 output channels split into split0=16 and split1=32.
    Grid = ceil(144 / 16) = 9 programs per patch × 4 patches = 36 total.
    """
    PATCHES  = 4
    BLOCK_HW = 16   # spatial positions per program (constexpr for tl.arange)
    SPLIT0   = 16   # first split  (16 output channels → out_ptr0)
    SPLIT1   = 32   # second split (32 output channels → out_ptr1)
    C_OUT    = SPLIT0 + SPLIT1   # = 48
    H_IN     = 16
    W_IN     = 16
    PAD      = 2
    HW       = 144  # 12 * 12 spatial positions in each unfolded window

    prog    = tl.program_id(0)
    n_hw    = tl.cdiv(HW, BLOCK_HW)
    patch   = prog // n_hw        # 0..3
    s_blk   = prog %  n_hw        # 0..8

    s_offs  = s_blk * BLOCK_HW + tl.arange(0, BLOCK_HW)   # spatial indices 0..143
    s_mask  = s_offs < HW

    # Decompose spatial index into (local_i, local_j) in the 12×12 window
    li_vec  = s_offs // 12   # 0..11
    lj_vec  = s_offs %  12   # 0..11

    # Decompose patch into (patch_i, patch_j) ∈ {0,1} × {0,1}
    pi      = patch // 2
    pj      = patch %  2

    # Global input coordinates  (H_PAD = W_PAD = 20)
    # Valid range [PAD, H_IN+PAD) = [2, 18);  anything outside is zero.
    hi = PAD + li_vec + pi * 8   # [BLOCK_HW]
    hj = PAD + lj_vec + pj * 8   # [BLOCK_HW]

    # NCHW base offset for this (patch_i, patch_j) tile
    base_x = (pi * 8 - PAD) * W_IN + (pj * 8 - PAD)    # scalar offset

    # Channel offsets for split0 and split1
    c0_offs = tl.arange(0, SPLIT0)   # [16]
    c1_offs = tl.arange(0, SPLIT1)   # [32]

    # Accumulators (float32 for precision)
    acc0 = tl.zeros([BLOCK_HW, SPLIT0], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_HW, SPLIT1], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K

        # ── Load input block: x[0, k, hi, hj] for all s_offs, k_offs ────────
        x_addrs = base_x + hi * W_IN + hj + k_offs[None, :] * (H_IN * W_IN)
        x_mask2 = s_mask[:, None] & k_mask[None, :]
        x_val   = tl.load(x_ptr + x_addrs, mask=x_mask2, other=0.0)  # [BLOCK_HW, BLOCK_K]

        # ── Load weight block: w[c_out, k, 0, 0] = w_ptr + c_out*K + k ─────
        # split0 part (c_out = 0..15)
        wc0   = c0_offs[:, None] * K + k_offs[None, :]                # [16, BLOCK_K]
        m0    = k_mask[None, :] & (c0_offs < SPLIT0)[:, None]
        w0    = tl.load(w_ptr + wc0, mask=m0, other=0.0)              # [16, BLOCK_K]

        # split1 part (c_out = 16..47)
        wc1   = (c1_offs + SPLIT0)[:, None] * K + k_offs[None, :]    # [32, BLOCK_K]
        m1    = k_mask[None, :] & (c1_offs < SPLIT1)[:, None]
        w1    = tl.load(w_ptr + wc1, mask=m1, other=0.0)              # [32, BLOCK_K]

        # ── GEMM: [BLOCK_HW, BLOCK_K] × [BLOCK_K, *] → [BLOCK_HW, *] ───────
        acc0 += tl.dot(x_val, tl.trans(w0))   # [BLOCK_HW, 16]
        acc1 += tl.dot(x_val, tl.trans(w1))   # [BLOCK_HW, 32]

    # ── Store out0: layout [8, 4, 16, 4] ──────────────────────────────────
    # out0[batch, pi, c0, pj]  →  strides [256, 64, 4, 1]
    #   addr = batch*256 + pi*64 + c0*4 + pj
    batch     = patch * n_hw + s_blk
    out0_base = batch * 256 + pi * 64 + pj   # scalar
    c0_addrs2 = out0_base + c0_offs * 4       # [16]
    tl.store(out_ptr0 + c0_addrs2, acc0[:, :].to(x_ptr.dtype.element_ty))

    # ── Store out1: layout [8, 4, 4, 32] ──────────────────────────────────
    # out1[batch, pi, pj, c1]  →  strides [512, 128, 32, 1]
    #   addr = batch*512 + pi*128 + pj*32 + c1
    out1_base = batch * 512 + pi * 128 + pj * 32
    c1_addrs2 = out1_base + c1_offs            # [32]
    tl.store(out_ptr1 + c1_addrs2, acc1[:, :].to(x_ptr.dtype.element_ty))


# ── Wrapper ───────────────────────────────────────────────────────────────────

@torch.fx.wrap
def fused_conv_pad_unfold_48_32(in_0, in_1):
    """
    in_0 : weight [384, 256, 1, 1]   (conv input: in_1, weight: in_0)
    in_1 : input  [1, 256, 16, 16]
    returns: (out0=[8,4,16,4],  out1=[8,4,4,32])
    """
    SPLIT0 = 16
    SPLIT1 = 32
    C_OUT  = SPLIT0 + SPLIT1   # = 48
    K      = in_1.shape[1]
    H_IN   = 16
    W_IN   = 16
    PAD    = 2
    HW     = 144  # 12 * 12

    out0 = torch.empty((8, 4, 16, 4), dtype=in_1.dtype, device=in_1.device)
    out1 = torch.empty((8, 4, 4, 32), dtype=in_1.dtype, device=in_1.device)

    BLOCK_HW = 16
    n_hw = (HW + BLOCK_HW - 1) // BLOCK_HW  # = 9

    # Grid: one program per (patch, spatial-block)
    grid = lambda meta: (4 * n_hw,)

    _fused_conv_patch_kernel_48_32[grid](
        in_1, in_0, out0, out1,
        K,
    )

    return (out0, out1)


# ── Pattern / replacement API ─────────────────────────────────────────────────

def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2  = torch.nn.functional.pad(conv2d, [2, 2, 2, 2], 'constant', None)
    tmp_3  = tmp_2.unfold(2, 12, 8)
    tmp_4  = tmp_3.unfold(3, 12, 8)
    tmp_5  = tmp_4.reshape(8, 48, 4, -1)
    tmp_6  = tmp_5.permute(0, 2, 3, 1)
    split  = torch.functional.split(tmp_6, [16, 32], dim=-1)
    tmp_8  = split[0]
    tmp_9  = split[1]
    tmp_10 = tmp_8.transpose(-1, -2)
    return (tmp_10, tmp_9)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fused_conv_pad_unfold_48_32