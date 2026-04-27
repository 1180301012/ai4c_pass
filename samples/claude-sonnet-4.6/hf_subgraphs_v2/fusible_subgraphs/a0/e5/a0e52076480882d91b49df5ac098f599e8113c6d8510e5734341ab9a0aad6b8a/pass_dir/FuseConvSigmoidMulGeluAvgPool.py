import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Full-fusion Triton kernel.
#   1×1 conv  →  sigmoid  →  broadcast_mul(in2)  →  GELU  →  avg_pool
# Dropout(p=0) is a no-op; flatten handled by output shape [B, OC].
#
# BLOCK_HW=64 is used for ALL spatial sizes:
#   HW=49 (7×7):   1 iteration, 49/64 = 77% utilisation
#   HW=64 (8×8):   1 iteration, 64/64 = 100% utilisation
#   HW=144 (12×12):3 iterations, 144/192 = 75% utilisation
# This beats larger BLOCK_HW values whose single-pass waste is worse.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        # BLOCK_HW=64 for all spatial sizes (77%/100%/75% utilisation).
        # stages=2 compiles fast to anchor autotune; stages=3 helps pipelining.
        # num_warps=2: 32 blocks/SM → 1 GPU wave for B=1, good for small batch.
        # num_warps=4: better MLP (memory-level parallelism) for B=32/128.
        triton.Config({'BLOCK_HW': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_HW': 64}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 64}, num_warps=4, num_stages=3),
    ],
    key=['B', 'OC', 'HW'],
)
@triton.jit
def _fused_full_kernel(
    in3_ptr,    # [B, IC, 1, 1]  SE input   (strides: IC, 1, 1, 1)
    weight_ptr, # [OC, IC, 1, 1] conv weight (strides: IC, 1, 1, 1)
    bias_ptr,   # [OC]
    in2_ptr,    # [B, OC, H, W]  feature map (contiguous)
    out_ptr,    # [B, OC]        output
    B, OC, HW,
    IC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    """
    Per (b, oc):
      conv  = dot(in3[b, :], weight[oc, :]) + bias[oc]
      scale = sigmoid(conv)
      out   = mean_{hw}( GELU( scale * in2[b, oc, hw] ) )
    """
    pid = tl.program_id(0)
    b  = pid // OC
    oc = pid % OC

    # ---- 1×1 conv as a dot product over IC channels ----
    ic_offs  = tl.arange(0, IC)
    x3       = tl.load(in3_ptr    + b  * IC + ic_offs).to(tl.float32)
    w        = tl.load(weight_ptr + oc * IC + ic_offs).to(tl.float32)
    bias_val = tl.load(bias_ptr   + oc).to(tl.float32)
    conv_out = tl.sum(x3 * w) + bias_val

    # ---- Sigmoid ----
    scale = 1.0 / (1.0 + tl.exp(-conv_out))

    # ---- Stream in2, apply GELU, accumulate average ----
    in2_base = (b * OC + oc) * HW
    acc = 0.0

    for hw_start in range(0, HW, BLOCK_HW):
        hw_offs = hw_start + tl.arange(0, BLOCK_HW)
        mask    = hw_offs < HW
        x2      = tl.load(in2_ptr + in2_base + hw_offs,
                          mask=mask, other=0.0).to(tl.float32)
        val     = scale * x2
        # Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
        gelu_v  = 0.5 * val * (1.0 + tl.math.erf(val * 0.7071067811865476))
        acc     = acc + tl.sum(gelu_v)

    tl.store(out_ptr + pid, (acc / HW).to(out_ptr.dtype.element_ty))


# ---------------------------------------------------------------------------
# @torch.fx.wrap wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def _fused_full_replacement(in_0, in_1, in_2, in_3):
    """
    in_0 : bias   [OC]
    in_1 : weight [OC, IC, 1, 1]
    in_2 : feat   [B, OC, H, W]
    in_3 : se_in  [B, IC, 1, 1]
    """
    B  = in_3.shape[0]
    IC = in_3.shape[1]
    OC = in_1.shape[0]
    H  = in_2.shape[2]
    W  = in_2.shape[3]
    HW = H * W

    out = torch.empty((B, OC), dtype=in_2.dtype, device=in_2.device)

    _fused_full_kernel[(B * OC,)](
        in_3, in_1, in_0, in_2, out,
        B, OC, HW, IC=IC,
    )

    return out


# ---------------------------------------------------------------------------
# Pattern / replacement interface
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3  = conv2d.sigmoid()
    tmp_4  = in_2 * tmp_3
    tmp_5  = torch.nn.functional.gelu(tmp_4, approximate='none')
    tmp_6  = torch.nn.functional.adaptive_avg_pool2d(tmp_5, 1)
    tmp_7  = tmp_6.flatten(1, -1)
    tmp_8  = torch.nn.functional.dropout(tmp_7, 0.0, False, False)
    return tmp_8


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return _fused_full_replacement