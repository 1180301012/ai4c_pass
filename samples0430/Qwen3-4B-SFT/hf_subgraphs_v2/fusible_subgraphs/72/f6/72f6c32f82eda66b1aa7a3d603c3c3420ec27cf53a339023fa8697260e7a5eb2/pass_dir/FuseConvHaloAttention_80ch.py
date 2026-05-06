"""
Pass: FuseConvHaloAttention_80ch
Matches the halo-attention self-attention pattern for Cout=80 (graph 9, float16).

Pattern:
  conv2d(in_1, in_0) → pad [2,2,2,2] → unfold(2,12,8) → unfold(3,12,8)
  → reshape(8,80,4,-1) → permute(0,2,3,1) → split([16,64], dim=-1)
  → [transpose(first 16), second 64]   ← returns (tmp_10, tmp_9)

The 1x1 conv is a GEMM:  X[1, Cin, 16, 16] @ W[Cout=80, Cin, 1, 1] → [1, 80, 16, 16]
Then the post-conv chain is a scatter-write into two shaped outputs:
  tmp_10 [8, 4, 16, 4]  (first 16 channels, transposed → tips)
  tmp_9  [8, 4, 4, 64]  (last 64  channels             → non-tips)
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
    tmp_5 = tmp_4.reshape(8, 80, 4, -1)
    tmp_6 = tmp_5.permute(0, 2, 3, 1)
    split  = torch.functional.split(tmp_6, [16, 64], dim=-1)
    tmp_8  = split[0]
    tmp_9  = split[1]
    tmp_10 = tmp_8.transpose(-1, -2)
    return (tmp_10, tmp_9)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

@triton.jit
def _fused_conv_scatter_kb16_s32(
    weight_ptr,   # [Cout=80, Cin, 1, 1] contiguous  → viewed as [80, Cin]
    x_ptr,        # [1, Cin, 16, 16]      contiguous  → viewed as [Cin, 256]
    out_tipped_ptr, # [8, 4, 16, 4] = 512 elements
    out_non_tipped_ptr, # [8, 4, 4, 64] = 8192 elements
    Cin: tl.constexpr,   # input channels (256 or 512)
    Cout: tl.constexpr,  # 80
    OW: tl.constexpr,    # 4  (also = KB / 16)
):
    """
    Grid: (8 * (HW // BLOCK_HW),)
    Each program handles BLOCK_HW spatial positions for ONE output bucket.

    Index decode (for graph 9: KB=16, OW=4):
      out_idx = b * 64 + j * 16 + col
      oh2 = out_idx // 64          (128 decoded from out_idx)
      j   = (out_idx // 16) % 4   (16 decoded from out_idx)
      col = out_idx % 16           (16 decoded from out_idx)

    ow2 = col >> 2  (= col // 4)
    Apply zero-padding: oh = oh2*8 + j - 2 (negative when oh2==0 or j<2)
    Apply zero-padding: ow = ow2*8 + col - 2 (negative when ow2==0)
    
    Inner loop over Cin in KB=16 tile steps.
    """
    KB = 16
    BLOCK_HW = 32

    pid = tl.program_id(0)

    # --- decode bucket and spatial tile indices ---
    nbins = 8   # 2 * 2 * 2 = 8 output buckets
    hw_tile_idx = pid % (HW // BLOCK_HW)
    buck = pid // (HW // BLOCK_HW)

    # spatial base within the 16x16 grid
    hw_base = hw_tile_idx * BLOCK_HW
    hw_off  = tl.arange(0, BLOCK_HW) % HW       # 0..HW-1

    # index into conv output (before scatter)
    m_vals   = buck * HW + hw_base + hw_off      # [BLOCK_HW]  position in [0, HW)

    oh2 = buck >> 2        # bucket oh-index: 0,1,2,3
    ow2 = (buck & 3) << 1  # bucket ow-index: 0,0,1,1,2,2,3,3

    j      = hw_off // 4   # tile position: 0..3
    col    = hw_off % 4    # local offset: 0..3
    ow     = ow2 * 8 + col - 2
    oh     = oh2 * 8 + j   - 2

    # --- accumulate GEMM in fp32 for numerical correctness ---
    acc = tl.zeros([BLOCK_HW, KB], dtype=tl.float32)

    Cin_tiles  = Cin // KB
    for k in range(0, Cin_tiles):           # KB=16 tiles along Cin
        k_off = tl.arange(0, KB)
        k_idx = k * KB + k_off              # [KB]  input-channel indices

        # weight tile  [KB, KB]:  W[:, k_start:k_start+KB]
        # weight[c, k]  →  weight_ptr + c * Cin + k
        w_ptrs = weight_ptr + (k + k_off[None, :]) * Cin + k_off[:, None]
        w_mask = k_off[None, :] < Cin
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # input tile  [KB, BLOCK_HW]:  X[hw_range, k_start:k_start+KB]
        # x[cin, hw]  →  x_ptr + cin * HW + hw
        x_ptrs = x_ptr + k_idx[:, None] * HW + m_vals[None, :]
        x_mask = k_idx[:, None] < Cin
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        acc += tl.dot(x, tl.trans(w))          # [BLOCK_HW, KB], fp32

    # Convert to element dtype (fp16 / bf16 / fp32)
    acc_f = acc.to(x.dtype)

    # --- write scattered results ---
    # (buck, oh, ow) encode m_val:  m_val = buck * 16 + oh * 16 + ow
    m_val    = buck * 16 + oh[:, None] * 16 + ow[None, :]
    KH       = OW    # = 4
    SHIFTS   = KB.bit_length()   # = 4
    OHW niño = 16    # = OW * OW = 16

    out_idx   = buck * (4 * KB) + j[None, :] * KB + col[None, :]
    soh   = (m_val // (OHW niño)) * KH   # = m_val // 16 * 4
    sow   = (m_val %  OHW niño) % (KH * 4) # = m_val % 16;  inner = m_val % 16
    sow   = (sow // OW) * OW + (sow % OW)  # round to nearest OW-multiple
    out_tidx = soh * (KB * 4) + sow

    # tips  (c < 16)
    tips_mask = (k + tl.arange(0, KB))[:, None] < KH
    tl.store(
        out_tipped_ptr + out_tidx[:, None] * KB + (k + tl.arange(0, KB))[None, :],
        acc_f,
        mask=tips_mask,
    )

    # non-tips (c >= KH)
    nt_offset = k * KB + tl.arange(0, KB)
    nt_mask   = (k + tl.arange(0, KB))[:, None] < Cout
    tl.store(
        out_non_tipped_ptr + out_tidx[:, None] * Cout + nt_offset[None, :],
        acc_f,
        mask=nt_mask,
    )


@triton.jit
def _post_gemm_kb16_s32(
    gemm_out_ptr,   # [HW=256, Cout=80] flattened row-major (same as conv2d output)
    out_tipped_ptr, # [8, 4, 16, 4]
    out_non_tipped_ptr, # [8, 4, 4, 64]
    Cout:  tl.constexpr,   # 80
    KB:    tl.constexpr,   # 16
    BLOCK_HW: tl.constexpr, # 32
):
    """
    Grid: (8 * (HW // BLOCK_HW),)
    Takes the flat reshaped matmul output [HW, Cout] and scatters each element
    to its correct position in the final (permuted+split) output tensors.
    """
    KB = 16
    BLOCK_HW = 32

    pid = tl.program_id(0)
    num_hw_tiles = HW // BLOCK_HW
    buck      = pid // num_hw_tiles
    hw_tile   = pid %  num_hw_tiles
    hw_base   = hw_tile * BLOCK_HW
    hw_off    = tl.arange(0, BLOCK_HW) % HW
    m_vals    = hw_base + hw_off   # same mapping as before

    # decode from final [8, 4, 16, 4] index
    # out_idx = b*(4*16) + j*16 + col  in [0, 512)
    i_1  = m_vals % (4 * KB)   # m_val maps to out_idx for ow2=0
    i_2  = (i_1 // KB) % 4
    i_3  = (i_1 // (KB * 4)) * (KB * 4) + (i_1 % KB)  # = i_1
    i_4  = i_1 % KB
    # recover oc from m_val
    oc    = (4 * KB + m_vals % (4 * KB)) % Cout   # = (4*16 + m_val) % 80
    mapped_oc = (oc // KB) * KB + (m_vals % KB)  # = oc (if KB|Cout-fit)

    # read matmul output  gemm_out[m, oc]  where  m = hw_base + hw_off
    g_ptrs = gemm_out_ptr + m_vals [:, None] * Cout + (oc[:, None])
    g_mask = oc[:, None] < Cout
    gather = tl.load(g_ptrs, mask=g_mask, other=0.0)

    # tips
    tips_mask = oc[:, None] < KH
    soh2 = (m_vals // (OHW niño)) * KH
    # sow2 = (m_vals % OHW niño) % (KH*4)
    ow2  = (m_vals % OHW niño) % KH
    so2  = (ow2 // OW) * OW + (ow2 % OW)
    out_t2idx = soh2 * (KB * 4) + so2
    tl.store(
        out_tipped_ptr + out_t2idx[:, None] * KB + (oc[:, None]),
        gather,
        mask=tips_mask,
    )

    # non-tips
    nt_mask = (oc[:, None] >= KH)
    tl.store(
        out_non_tipped_ptr + out_t2idx[:, None] * Cout + oc[:, None],
        gather,
        mask=nt_mask,
    )


# ---------------------------------------------------------------------------
# Kernel wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_conv_halo_80ch(in_0, in_1):
    """
    in_0 : [Cout=80,  Cin, 1, 1]  weight (may be on CPU)
    in_1 : [1, Cin, 16, 16]        feature map (on CUDA)

    Returns: (tmp_10, tmp_9)  matching the model.py outputs
    tmp_10 : [8, 4, 16, 4]
    tmp_9  : [8, 4, 4, 64]
    """
    Cin   = in_0.shape[1]
    Cout  = in_0.shape[0]
    HW    = 256   # 16 * 16

    # handle CPU weight
    w     = in_0.to(in_1.device)

    # Build flat views
    x_flat     = in_1.reshape(Cin, HW)   # [Cin, 256]
    weight_flat = w.reshape(Cout, Cin)    # [80, Cin]

    out_tipped  = torch.empty((8, 4, 16, 4),    dtype=in_1.dtype, device=in_1.device)
    out_non_tipped = torch.empty((8, 4, 4, 64), dtype=in_1.dtype, device=in_1.device)

    # --- kernel 1: GEMM + scatter write in one pass ---
    BLOCK_HW   = 32
    KB         = 16
    num_hw_tiles = HW // BLOCK_HW   # = 8
    grid_gemm  = (8 * num_hw_tiles,)  # = 64

    _fused_conv_scatter_kb16_s32[grid_gemm](
        weight_flat,
        x_flat,
        out_tipped,
        out_non_tipped,
        Cin=Cin,
        Cout=80,
        OW=4,
    )

    return out_tipped, out_non_tipped