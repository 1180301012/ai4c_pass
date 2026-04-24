import torch
import triton
import triton.language as tl


# ── Pattern ──────────────────────────────────────────────────────────────────
def pattern(in_0, in_1, in_2, in_3, in_4):
    tmp_4 = in_4.reshape(1, 512, 16, 16)
    tmp_5 = torch.nn.functional.avg_pool2d(tmp_4, 2, 2, 0, False, True, None)
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_7 = torch.nn.functional.silu(tmp_6, inplace=True)
    return tmp_7


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


# ── Fused Triton kernel ───────────────────────────────────────────────────────
#
# The kernel fuses:
#   reshape([4,128,256] → [1,512,16,16])  (view – free, no copy)
#   avg_pool2d(2, stride=2, pad=0, count_include_pad=True)
#   batch_norm  (inference; training=False)
#   silu
#
# Grid: 1-D over all output elements, tiled by BLOCK_SIZE.
# Output shape: [1, 512, 8, 8]  →  C=512, H_out=8, W_out=8, N=32768
#
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=[],
)
@triton.jit
def _fused_kernel(
    in4_ptr,        # [4, 128, 256]  contiguous, bfloat16 / float16
    mean_ptr,       # [C]  running_mean
    var_ptr,        # [C]  running_var
    weight_ptr,     # [C]  gamma
    bias_ptr,       # [C]  beta
    out_ptr,        # [1, 512, 8, 8] output
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    N = 512 * 8 * 8          # 32768
    mask = offsets < N

    # ── decode flat index ──────────────────────────────────────────────────
    # offsets = c * 64 + oh * 8 + ow
    c  = offsets // 64
    hw = offsets % 64
    oh = hw // 8
    ow = hw % 8

    # ── avg_pool2d: gather 4 input values ──────────────────────────────────
    # The input viewed as [1,512,16,16] has strides [131072, 256, 16, 1].
    # Its flat address for channel c at input row h, col w equals
    #   c*256 + h*16 + w  =  c*256 + (oh*2)*16 + (ow*2)
    # which is also the flat address in the original [4,128,256] layout.
    in_base  = c * 256 + oh * 32 + ow * 2   # top-left corner
    addr00   = in_base
    addr01   = in_base + 1
    addr10   = in_base + 16
    addr11   = in_base + 17

    x00 = tl.load(in4_ptr + addr00, mask=mask, other=0.0).to(tl.float32)
    x01 = tl.load(in4_ptr + addr01, mask=mask, other=0.0).to(tl.float32)
    x10 = tl.load(in4_ptr + addr10, mask=mask, other=0.0).to(tl.float32)
    x11 = tl.load(in4_ptr + addr11, mask=mask, other=0.0).to(tl.float32)
    pooled = (x00 + x01 + x10 + x11) * 0.25

    # ── batch_norm (inference) ─────────────────────────────────────────────
    mean  = tl.load(mean_ptr   + c, mask=mask, other=0.0).to(tl.float32)
    var   = tl.load(var_ptr    + c, mask=mask, other=1.0).to(tl.float32)
    gamma = tl.load(weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
    beta  = tl.load(bias_ptr   + c, mask=mask, other=0.0).to(tl.float32)

    scale  = gamma / tl.sqrt(var + 1e-5)
    bn_val = scale * pooled + (beta - scale * mean)

    # ── silu activation ────────────────────────────────────────────────────
    silu_val = bn_val * tl.sigmoid(bn_val)

    # ── store ──────────────────────────────────────────────────────────────
    tl.store(out_ptr + offsets, silu_val.to(out_ptr.dtype.element_ty), mask=mask)


# ── Wrapper (must be @torch.fx.wrap) ─────────────────────────────────────────
@torch.fx.wrap
def fused_reshape_avgpool_bn_silu(in_0, in_1, in_2, in_3, in_4):
    """
    in_0 : running_mean  [512]
    in_1 : running_var   [512]
    in_2 : bias (beta)   [512]
    in_3 : weight (γ)    [512]
    in_4 : activation    [4, 128, 256]
    """
    out = torch.empty((1, 512, 8, 8), dtype=in_4.dtype, device=in_4.device)

    N = 512 * 8 * 8   # 32768
    grid = lambda meta: ((N + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    _fused_kernel[grid](
        in_4,       # input activation
        in_0,       # running_mean
        in_1,       # running_var
        in_3,       # weight (gamma)
        in_2,       # bias    (beta)
        out,
    )

    return out


# ── Replacement entry point ───────────────────────────────────────────────────
def replacement_func():
    return fused_reshape_avgpool_bn_silu