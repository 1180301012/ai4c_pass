import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: einsum('bchw,bchj->bhwj', in_2, in_1) + cat + softmax + slice
# Both tmp_3 (full softmax) and tmp_4 (first 64 cols) are returned by the model.
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2):
    einsum = torch.functional.einsum('bchw,bchj->bhwj', in_2, in_1)
    tmp_2 = torch.cat([in_0, einsum], dim=-1)
    tmp_3 = torch.nn.functional.softmax(tmp_2, dim=-1)
    tmp_4 = tmp_3[(Ellipsis, slice(None, 64, None))]
    return (tmp_3, tmp_4)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ---------------------------------------------------------------------------
# Triton kernel
#
# Grid = (B * C,)
# Each program (pid) owns one (b, c) pair.
#
# For that (b, c):
#   1. Load in_2[b,c,:,:]    shape [HW, C]  -> [BLOCK_HW, BLOCK_C]
#   2. Load in_1[b,c,:,:]    shape [C, HW]   -> [BLOCK_C, BLOCK_HW]
#   3. matvec: result[w] = sum_k(in2[w,k] * in1[k,w_target])
#      Actually: einsum result has einsum[b,c,h,w] = sum_k in2[b,c,h,k]*in1[b,c,k,w]
#      For each h (row) we compute 64 dot products of length 64.
#   4. Load in_0[b,c,h,:]   shape [HW, HW]  -> [BLOCK_HW, BLOCK_HW]
#   5. cat: x[b,c,h,:] = [in0[b,c,h,:64], matvec[b,c,h,:]]  shape [128]
#   6. softmax over 128 elements
#   7. Store out_full[b,c,h,:]               shape [128]
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64, 'BLOCK_C': 64}, num_warps=4),
        triton.Config({'BLOCK_HW': 64, 'BLOCK_C': 64}, num_warps=8),
        triton.Config({'BLOCK_HW': 64, 'BLOCK_C': 64}, num_warps=2),
    ],
    key=['B', 'C', 'HW'],
)
@triton.jit
def fused_einsum_cat_softmax_kernel(
    in0_ptr, in1_ptr, in2_ptr,
    out_full_ptr,
    B, C, HW,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // C
    c = pid % C

    hw_offs = tl.arange(0, BLOCK_HW)   # [0 .. HW-1]  (HW == BLOCK_HW == 64)
    c_offs  = tl.arange(0, BLOCK_C)    # [0 .. C-1]   (C  == BLOCK_C == 64)

    HW_C = HW * C
    # Base offset for this (b, c) in any [B, C, *, *] tensor
    base = b * (C * HW * HW) + c * (HW * HW)

    # ------------------------------------------------------------------
    # Load in_2[b, c, :, :]  shape [HW, C]  -> [BLOCK_HW, BLOCK_C]
    # ------------------------------------------------------------------
    in2 = tl.load(
        in2_ptr + base + hw_offs[:, None] * C + c_offs[None, :],
        mask=(hw_offs[:, None] < HW) & (c_offs[None, :] < C),
        other=0.0,
    )  # float16/bfloat16

    # ------------------------------------------------------------------
    # Load in_1[b, c, :, :]  shape [C, HW]  -> [BLOCK_C, BLOCK_HW]
    # ------------------------------------------------------------------
    in1 = tl.load(
        in1_ptr + base + c_offs[:, None] * HW + hw_offs[None, :],
        mask=(c_offs[:, None] < C) & (hw_offs[None, :] < HW),
        other=0.0,
    )  # float16/bfloat16

    # ------------------------------------------------------------------
    # Matvec:  result[w] = sum_k( in2[w,k] * in1[k,w_target] )
    # But note: einsum is over k (C dim), and we need result[h, w] =
    #   sum_k( in2[b,c,h,k] * in1[b,c,k,w] )
    # For every h row independently.
    # in2: [BLOCK_HW, BLOCK_C],  in1: [BLOCK_C, BLOCK_HW]
    # ------------------------------------------------------------------
    in2_f32 = in2.to(tl.float32)
    in1_f32 = in1.to(tl.float32)

    matvec = tl.sum(in2_f32[:, :, None] * in1_f32[None, :, :], axis=1)
    # shape: [BLOCK_HW, BLOCK_HW]  (h and w both index 0..HW-1)

    # ------------------------------------------------------------------
    # Load in_0[b, c, :, :]  shape [HW, HW]  -> [BLOCK_HW, BLOCK_HW]
    # ------------------------------------------------------------------
    in0 = tl.load(
        in0_ptr + base + hw_offs[:, None] * HW + hw_offs[None, :],
        mask=(hw_offs[:, None] < HW) & (hw_offs[None, :] < HW),
        other=0.0,
    )  # float16/bfloat16

    in0_f32 = in0.to(tl.float32)

    # ------------------------------------------------------------------
    # Cat along last dim (128-wide row for each h):
    #   x[h, j] = in0[b,c,h,j]          for j in 0..HW-1
    #           = matvec[b,c,h,j-HW]      for j in HW..2*HW-1
    # ------------------------------------------------------------------
    # Allocate 128-wide rows
    x = tl.zeros((BLOCK_HW, 2 * BLOCK_HW), dtype=tl.float32)

    # First 64 columns: in_0
    x = tl.where(
        tl.arange(0, 2 * BLOCK_HW)[None, :] < BLOCK_HW,
        in0_f32,
        x,
    )
    # Next 64 columns: matvec  (matvec[h, w] -> x[h, HW+w])
    x = tl.where(
        tl.arange(0, 2 * BLOCK_HW)[None, :] >= BLOCK_HW,
        matvec,
        x,
    )

    # ------------------------------------------------------------------
    # Softmax over 128-element rows
    # ------------------------------------------------------------------
    x_max    = tl.max(x, axis=1)                       # [BLOCK_HW]
    x_exp    = tl.exp(x - x_max[:, None])              # [BLOCK_HW, 2*BLOCK_HW]
    x_sum    = tl.sum(x_exp, axis=1)                  # [BLOCK_HW]
    x_sm     = x_exp / x_sum[:, None]                 # [BLOCK_HW, 2*BLOCK_HW]

    # ------------------------------------------------------------------
    # Store out_full[b, c, :, :]   (128 columns)
    # out_full shape: [B, C, H, 128]  -> stride [C*H*128, H*128, 128, 1]
    # For (b,c) block at base: in [B, C, H, 128], base = b*C*H*128 + c*H*128
    # ------------------------------------------------------------------
    out_base = b * (C * HW * 2) + c * (HW * 2)
    tl.store(
        out_full_ptr + out_base + hw_offs[:, None] * (2 * HW) + hw_offs[None, :],
        x_sm.to(in0.dtype),
        mask=(hw_offs[:, None] < HW) & (hw_offs[None, :] < HW),
    )


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_einsum_cat_softmax(in_0, in_1, in_2):
    B, C, H, W = in_0.shape
    # in_1, in_2 shape: [B, C, H, W] (= [B, 64, 64, 64])
    # Output: [B, C, H, 2*W]  (= [B, 64, 64, 128])
    out_full = torch.empty((B, C, H, W * 2), dtype=in_0.dtype, device=in_0.device)

    grid = (B * C,)
    fused_einsum_cat_softmax_kernel[grid](
        in_0, in_1, in_2,
        out_full,
        B, C, H,
    )

    # out_full[..., :W] is a zero-copy view — same memory as first W cols of out_full
    out_slice = out_full[(Ellipsis, slice(None, W, None))]
    return (out_full, out_slice)


def replacement_func():
    return fused_einsum_cat_softmax