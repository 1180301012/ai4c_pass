"""
Pass: FuseConvHaloAttention_48ch
Matches the halo-attention self-attention pattern for Cout=48 (graph 10, bfloat16/float32).

Pattern:
  conv2d(in_1, in_0) → pad [2,2,2,2] → unfold(2,12,8) → unfold(3,12,8)
  → reshape(8,48,4,-1) → permute(0,2,3,1) → split([16,32], dim=-1)
  → [transpose(first 16), second 32]   ← returns (tmp_10, tmp_9)

The 1x1 conv is a GEMM:  X[1, Cin, 16, 16] @ W[Cout=48, Cin, 1, 1] → [1, 48, 16, 16]
Then the post-conv chain is a scatter-write into two shaped outputs:
  tmp_10 [8, 4, 16, 4]  (first KH=16 channels, transposed → tips)
  tmp_9  [8, 4, 4, 32]  (last anoi-KH=32 channels              → non-tips)
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern
# ---------------------------------------------------------------------------

def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.nn.functional.pad(conv2d, [2, 2, 2, 2], 'constant', None)
    tmp_3 = tmp_2.unfold(2, 12, 8)
    tmp_4 = tmp_3.unfold(3, 12, 8)
    tmp_5 = tmp_4.reshape(8, 48, 4, -1)
    tmp_6 = tmp_5.permute(0, 2, 3, 1)
    split  = torch.functional.split(tmp_6, [16, 32], dim=-1)
    tmp_8  = split[0]
    tmp_9  = split[1]
    tmp_10 = tmp_8.transpose(-1, -2)
    return (tmp_10, tmp_9)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ---------------------------------------------------------------------------
# Shared kernel (reused for both graphs; Triton compiles separate versions per
# constexpr value)
# ---------------------------------------------------------------------------

@triton.jit
def _fused_conv_scatter(
    weight_ptr,   # [Cout, Cin, 1, 1] contiguous  → viewed as [Cout, Cin]
    x_ptr,        # [1, Cin, 16, 16]      contiguous  → viewed as [Cin, 256]
    out_tipped_ptr,
    out_non_tipped_ptr,
    Cin:  tl.constexpr,
    Cout: tl.constexpr,
    KH:   tl.constexpr,    # first  chunk of channels → tmp_10, e.g. 4 or 16
    OW:   tl.constexpr,    # = 4
    KB:   tl.constexpr,    # = 16
):
    """
    Grid : (8 * (HW // BLOCK_HW),)
    Block: handles BLOCK_HW spatial positions for ONE output bucket.

    KB=16 always → KB.bit_length() = 4 gives the OW shift.
    KB=16 divides OW=4 (16 = 4*OW), so col → ow2: col // OW = col // 4.
    Between KB consecutive out_idx values (per bucket), OW=4 consecutive values
    share the same ow2 bucket row.
    """
    BLOCK_HW  = 32
    HW_TILES  = HW // BLOCK_HW                 # runtime (C int)

    pid         = tl.program_id(0)
    hw_tile_idx = pid % (HW_TILES)
    buck        = pid // (HW_TILES)

    hw_base = hw_tile_idx * BLOCK_HW
    hw_off  = tl.arange(0, BLOCK_HW) % HW
    m_vals  = buck * HW + hw_base + hw_off      # conv2d row index in [0, HW)

    # decode bucket coordinates
    oh2 = buck >> 2                    # tile-oh in {0,1,2,3}
    ow2 = (buck & 3) << 1              # tile-ow in {0,0,1,1,2,2,3,3}

    # decode hw_off → (j, col) in the OW×OW spatial tile
    j      = hw_off // OW              # tile position in 0..3 (per OW direction)
    col    = hw_off % OW               # local column in 0..3
    ow     = ow2 * 8 + col - 2        # input ow coordinate (zero-pad handled by mask)
    oh     = oh2 * 8 + j  - 2         # input oh coordinate (zero-pad handled by mask)

    acc = tl.zeros([BLOCK_HW, KB], dtype=tl.float32)

    KB_TILES = Cin // KB               # compile-time
    for k in range(0, KB_TILES):
        k_off = tl.arange(0, KB)
        k_idx = k * KB + k_off

        w_ptrs = weight_ptr + (k + k_off[None, :]) * Cin + k_off[:, None]
        w_mask = k_off[None, :] < Cin
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        x_ptrs = x_ptr + k_idx[:, None] * HW + m_vals[None, :]
        x_mask = k_idx[:, None] < Cin
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        acc += tl.dot(x, tl.trans(w))

    acc_f = acc.to(x.dtype)

    # ---- index decode for scatter ----
    KB_B = KB.bit_length()   # = 4  (since KB=16 = 2^4)
    OHW  = OW * OW            # = 16   (number of spatial positions per tile)

    m_val = buck * OHW + oh[:, None] * OW + ow[None, :]   # [BLOCK_HW]

    out_idx    = buck * (4 * KB) + j[None, :] * KB + col[None, :]
    soh        = (m_val // OHW) * KH   # bucket OHW × KH
    # sow = (m_val % OHW) % (KH*OW)
    # But since KB=16=OW*KH and (m_val%OHW)∈[0,15]:  for KH=4: (m%16)%64=m%16
    #                                            for KH=16: (m%16)%64=m%16
    # So sow = m_val % OHW = m_val % 16 (= col equivalently, since col = m_val%OW)
    # Actually: sow = (m_val % OHW) % (KH * OW)
    # = (m_val % 16) % (KH*4)  = m_val % 16  since KH*4 ≥ 16
    # So: ow2_idx = (col >> KB_B) * OW + (col & (1 << KB_B - 1))
    ow2_idx = (col >> KB_B) * OW + (col & ((1 << (KB_B - 1)) - 1))
    out_tidx = soh * (KB * OW) + ow2_idx                    # spatial key

    # tips: oc ∈ [0, KH)
    tips_mask = (k + tl.arange(0, KB))[:, None] < KH
    tl.store(
        out_tipped_ptr + out_tidx[:, None] * KB + (k + tl.arange(0, KB))[None, :],
        acc_f, mask=tips_mask,
    )

    # non-tips: oc ∈ [KH, Cout)
    nt_offset = k * KB + tl.arange(0, KB)
    nt_mask   = (k + tl.arange(0, KB))[:, None] < Cout
    tl.store(
        out_non_tipped_ptr + out_tidx[:, None] * Cout + nt_offset[None, :],
        acc_f, mask=nt_mask,
    )


# ---------------------------------------------------------------------------
# Kernel wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_conv_halo_48ch(in_0, in_1):
    """
    in_0 : [Cout=48, Cin, 1, 1]    weight
    in_1 : [1, Cin, 16, 16]        feature map (CUDA)

    Returns: (tmp_10, tmp_9)
      tmp_10 : [8, 4, KH=16, OW=4]     (first  16 channels, transposed)
      tmp_9  : [8, 4, OW=4, 32]        (last   32 channels, straight)
    """
    Cin   = in_0.shape[1]
    Cout  = in_0.shape[0]    # 48
    HW    = 256              # 16 * 16
    KH    = 16               # split[0]

    # Move weight to same device as feature map
    w       = in_0.to(in_1.device)
    x_flat   = in_1.reshape(Cin, HW)       # [Cin, 256]
    wt_flat  = w.reshape(Cout, Cin)         # [48, Cin]

    out_tipped      = torch.empty((8, 4, 16, 4),    dtype=in_1.dtype, device=in_1.device)
    out_non_tipped  = torch.empty((8, 4, 4, 32),    dtype=in_1.dtype, device=in_1.device)

    BLOCK_HW       = 32
    KB             = 16
    grid           = (8 * (HW // BLOCK_HW),)       # (304,)

    _fused_conv_scatter[grid](
        wt_flat,
        x_flat,
        out_tipped,
        out_non_tipped,
        Cin=Cin,
        Cout=48,
        KH=KH,
        OW=4,
        KB=KB,
    )

    return out_tipped, out_non_tipped