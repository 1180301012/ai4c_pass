import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: matches the ENTIRE forward pass (conv2d + all post-processing)
#
# in_0: bias    [C_out=1024]
# in_1: weight  [C_out=1024, C_in=64, 1, 1]
# in_2: activation [B, C_out=1024, H, W]
# in_3: x_se   [B, C_in=64, 1, 1]
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.sigmoid()
    tmp_4 = in_2 * tmp_3
    tmp_5 = torch.nn.functional.gelu(tmp_4, approximate='none')
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(tmp_5, 1)
    tmp_7 = tmp_6.flatten(1, -1)
    tmp_8 = torch.nn.functional.dropout(tmp_7, 0.0, False, False)
    return tmp_8


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# ---------------------------------------------------------------------------
# Triton kernel — fully fused 1x1-conv + sigmoid + mul + gelu + avgpool
#
# The 1x1 convolution with input [B, K, 1, 1] and weight [C, K, 1, 1]
# is mathematically equivalent to a GEMM: [B, K] @ [K, C] + bias[C].
#
# For each tile (b_block, c_block) of shape (BLOCK_B, BLOCK_C):
#   1. Dot-product  path (BLOCK_B==1): elementwise dot + bias
#   2. GEMM         path (BLOCK_B>=16): tl.dot for the conv
# Then for every (b, c) in the tile:
#   3. Load in_2[b, c, 0:HW] — BLOCK_HW elements (masked)
#   4. Compute avg{ gelu(in_2 * sigmoid(conv_result)) } over HW
#   5. Store output[b, c]
#
# Grid: (ceil(B/BLOCK_B), ceil(C/BLOCK_C))
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        # BLOCK_B=1: dot-product path, best for B=1
        triton.Config({'BLOCK_B': 1,  'BLOCK_C': 64},  num_warps=4),
        triton.Config({'BLOCK_B': 1,  'BLOCK_C': 64},  num_warps=8),
        triton.Config({'BLOCK_B': 1,  'BLOCK_C': 128}, num_warps=8),
        triton.Config({'BLOCK_B': 1,  'BLOCK_C': 128}, num_warps=16),
        # BLOCK_B=16: tl.dot path, good for medium B
        triton.Config({'BLOCK_B': 16, 'BLOCK_C': 32},  num_warps=4),
        triton.Config({'BLOCK_B': 16, 'BLOCK_C': 32},  num_warps=8),
        triton.Config({'BLOCK_B': 16, 'BLOCK_C': 64},  num_warps=8),
        triton.Config({'BLOCK_B': 16, 'BLOCK_C': 64},  num_warps=16),
        # BLOCK_B=32: tl.dot path, good for large B
        triton.Config({'BLOCK_B': 32, 'BLOCK_C': 32},  num_warps=8),
        triton.Config({'BLOCK_B': 32, 'BLOCK_C': 32},  num_warps=16),
        triton.Config({'BLOCK_B': 32, 'BLOCK_C': 64},  num_warps=16),
    ],
    key=['B', 'C', 'HW'],
)
@triton.jit
def _fully_fused_kernel(
    bias_ptr,    # [C]
    w_ptr,       # [C, K]   (weight reshaped from [C,K,1,1])
    x2_ptr,      # [B, C, H, W]
    x3_ptr,      # [B, K]   (x_se reshaped from [B,K,1,1])
    out_ptr,     # [B, C]
    B, C, K, HW,
    IS_FP16:  tl.constexpr,
    IS_BF16:  tl.constexpr,
    BLOCK_B:  tl.constexpr,
    BLOCK_C:  tl.constexpr,
    BLOCK_K:  tl.constexpr,   # = K = 64 (fixed)
    BLOCK_HW: tl.constexpr,   # next_power_of_2(HW)
):
    b_block = tl.program_id(0)
    c_block = tl.program_id(1)

    b_start = b_block * BLOCK_B
    c_start = c_block * BLOCK_C

    c_local = tl.arange(0, BLOCK_C)
    c_offs  = c_start + c_local        # [BLOCK_C]
    c_mask  = c_offs < C
    k_offs  = tl.arange(0, BLOCK_K)   # [K=64]

    inv_sqrt2 = 0.7071067811865476

    # ---- load weight tile [BLOCK_C, K] ----
    w = tl.load(
        w_ptr + c_offs[:, None] * K + k_offs[None, :],
        mask=c_mask[:, None],
        other=0.0
    ).to(tl.float32)   # [BLOCK_C, K]

    # ---- load bias tile [BLOCK_C] ----
    bias = tl.load(bias_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)

    if BLOCK_B == 1:
        # ---- Dot-product path (single batch row) ----
        b = b_start
        # Load x3[b, :] → [K]
        x3 = tl.load(x3_ptr + b * K + k_offs, mask=b < B, other=0.0).to(tl.float32)
        # Dot product: [K] · [BLOCK_C, K] → [BLOCK_C]
        # = sum(w * x3[None,:], axis=1)
        conv_out = tl.sum(w * x3[None, :], axis=1) + bias   # [BLOCK_C]
        sig = tl.sigmoid(conv_out)                           # [BLOCK_C]

        # Load x2[b, c_offs, 0:HW] → [BLOCK_C, BLOCK_HW]
        hw_local = tl.arange(0, BLOCK_HW)
        hw_mask  = hw_local < HW
        x_base   = b * C * HW + c_start * HW
        x2_off   = c_local[:, None] * HW + hw_local[None, :]  # [BLOCK_C, BLOCK_HW]
        x2_mask  = c_mask[:, None] & hw_mask[None, :]
        x2 = tl.load(x2_ptr + x_base + x2_off,
                     mask=x2_mask, other=0.0).to(tl.float32)  # [BLOCK_C, BLOCK_HW]

        val    = x2 * sig[:, None]
        gelu_v = val * 0.5 * (1.0 + tl.math.erf(val * inv_sqrt2))
        avg    = tl.sum(gelu_v, axis=1) / HW                  # [BLOCK_C]

        bc_offs = b * C + c_offs
        if IS_FP16:
            tl.store(out_ptr + bc_offs, avg.to(tl.float16), mask=c_mask)
        elif IS_BF16:
            tl.store(out_ptr + bc_offs, avg.to(tl.bfloat16), mask=c_mask)
        else:
            tl.store(out_ptr + bc_offs, avg.to(tl.float32), mask=c_mask)

    else:
        # ---- GEMM path (BLOCK_B rows at once) ----
        b_local = tl.arange(0, BLOCK_B)
        b_offs  = b_start + b_local    # [BLOCK_B]
        b_mask  = b_offs < B

        # Load x3[b_offs, :] → [BLOCK_B, K]
        x3 = tl.load(
            x3_ptr + b_offs[:, None] * K + k_offs[None, :],
            mask=b_mask[:, None],
            other=0.0
        ).to(tl.float32)   # [BLOCK_B, K]

        # GEMM: [BLOCK_B, K] × [K, BLOCK_C] = [BLOCK_B, BLOCK_C]
        conv_out = tl.dot(x3, tl.trans(w), allow_tf32=False)   # [BLOCK_B, BLOCK_C]
        conv_out = conv_out + bias[None, :]                      # broadcast bias
        sig = tl.sigmoid(conv_out)                               # [BLOCK_B, BLOCK_C]

        # Load x2 3D tile: [BLOCK_B, BLOCK_C, BLOCK_HW]
        hw_local = tl.arange(0, BLOCK_HW)
        hw_mask  = hw_local < HW
        # x2[b, c, hw] = x2_ptr + b*C*HW + c*HW + hw
        x2_offs = (b_offs[:, None, None] * C * HW
                   + c_offs[None, :, None] * HW
                   + hw_local[None, None, :])   # [BLOCK_B, BLOCK_C, BLOCK_HW]
        x2_mask = (b_mask[:, None, None]
                   & c_mask[None, :, None]
                   & hw_mask[None, None, :])
        x2 = tl.load(x2_ptr + x2_offs,
                     mask=x2_mask, other=0.0).to(tl.float32)  # [BLOCK_B, BLOCK_C, BLOCK_HW]

        val    = x2 * sig[:, :, None]
        gelu_v = val * 0.5 * (1.0 + tl.math.erf(val * inv_sqrt2))
        avg    = tl.sum(gelu_v, axis=2) / HW   # [BLOCK_B, BLOCK_C]

        out_offs = b_offs[:, None] * C + c_offs[None, :]  # [BLOCK_B, BLOCK_C]
        out_mask = b_mask[:, None] & c_mask[None, :]
        if IS_FP16:
            tl.store(out_ptr + out_offs, avg.to(tl.float16), mask=out_mask)
        elif IS_BF16:
            tl.store(out_ptr + out_offs, avg.to(tl.bfloat16), mask=out_mask)
        else:
            tl.store(out_ptr + out_offs, avg.to(tl.float32), mask=out_mask)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def _fully_fused_conv_gelu_avgpool(bias, weight, in_2, x_se):
    """
    Fuses: 1x1-conv(x_se, weight, bias) → sigmoid → * in_2 → gelu → avgpool → flatten

    bias   : [C]         (in_0)
    weight : [C, K, 1, 1] (in_1)
    in_2   : [B, C, H, W] (in_2)
    x_se   : [B, K, 1, 1] (in_3)
    returns: [B, C]
    """
    B  = in_2.shape[0]
    C  = in_2.shape[1]
    H  = in_2.shape[2]
    W  = in_2.shape[3]
    HW = H * W
    K  = x_se.shape[1]        # C_in = 64

    BLOCK_K  = K                          # = 64, entire K in one tile
    BLOCK_HW = triton.next_power_of_2(HW) # >= HW, power-of-2

    out = torch.empty((B, C), dtype=in_2.dtype, device=in_2.device)

    is_fp16 = (in_2.dtype == torch.float16)
    is_bf16 = (in_2.dtype == torch.bfloat16)

    def grid(META):
        return (
            triton.cdiv(B, META['BLOCK_B']),
            triton.cdiv(C, META['BLOCK_C']),
        )

    _fully_fused_kernel[grid](
        bias,   weight, in_2, x_se, out,
        B, C, K, HW,
        IS_FP16=is_fp16,
        IS_BF16=is_bf16,
        BLOCK_K=BLOCK_K,
        BLOCK_HW=BLOCK_HW,
    )

    return out


# ---------------------------------------------------------------------------
# replacement_func: zero-argument factory — return the callable, don't call it
# ---------------------------------------------------------------------------
def replacement_func():
    return _fully_fused_conv_gelu_avgpool