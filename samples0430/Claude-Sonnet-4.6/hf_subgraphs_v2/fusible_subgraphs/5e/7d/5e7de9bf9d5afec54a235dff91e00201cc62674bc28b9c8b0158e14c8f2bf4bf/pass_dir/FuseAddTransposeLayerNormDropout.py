import torch
import triton
import triton.language as tl

# ──────────────────────────────────────────────────────────────────────────────
# Fused kernel:
#   gelu(conv1d_out)  +  avg_pool1d(in_3)
#   → slice both to L=124  → add  → transpose(1,2)
#   → layer_norm  → identity dropout
#
# Input shapes (this graph):
#   conv_out : [1, 768, 125]  (raw conv1d output, stride_C = 125)
#   in_3     : [1, 768, 249]  (original hidden states, stride_C = 249)
#   weight   : [768]
#   bias     : [768]
#
# Output: [1, 124, 768]
#
# Kernel saves 3 separate PyTorch kernels: gelu, avg_pool1d, elementwise-add
# ──────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4,  num_stages=1),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8,  num_stages=1),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16, num_stages=1),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=32, num_stages=1),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16, num_stages=2),
    ],
    key=['N', 'L'],
)
@triton.jit
def _fused_gelu_avgpool_add_ln_kernel(
    conv_ptr,        # conv1d output [B, N, L+1=125] – gelu is applied here
    in3_ptr,         # in_3          [B, N, 2L+1=249]
    w_ptr,           # layer-norm weight [N]
    bias_ptr,        # layer-norm bias   [N]
    out_ptr,         # output [B, L, N]
    N,               # hidden dim (768)
    L,               # output seq len (124)
    eps,             # 1e-5
    conv_stride_B,   # conv_out.stride(0)
    conv_stride_N,   # conv_out.stride(1) = 125
    in3_stride_B,    # in_3.stride(0)
    in3_stride_N,    # in_3.stride(1) = 249
    BLOCK_SIZE: tl.constexpr,
):
    # ── Single row per program ────────────────────────────────────────────────
    row      = tl.program_id(0)            # 0 .. batch*L-1
    batch_id = row // L
    t        = row - batch_id * L

    offsets = tl.arange(0, BLOCK_SIZE)
    mask    = offsets < N

    conv_base = batch_id * conv_stride_B
    in3_base  = batch_id * in3_stride_B

    INV_SQRT2 = 0.7071067811865476

    # ── Load conv1d output and apply exact GELU ───────────────────────────────
    cv  = tl.load(conv_ptr + conv_base + offsets * conv_stride_N + t,
                  mask=mask, other=0.0)
    cvf = cv.to(tl.float32)
    g   = (cvf * 0.5 * (1.0 + tl.erf(cvf * INV_SQRT2))).to(cv.dtype)

    # ── avg_pool1d: (in_3[c, 2t] + in_3[c, 2t+1]) / 2 ───────────────────────
    a0  = tl.load(in3_ptr + in3_base + offsets * in3_stride_N + 2 * t,
                  mask=mask, other=0.0)
    a1  = tl.load(in3_ptr + in3_base + offsets * in3_stride_N + 2 * t + 1,
                  mask=mask, other=0.0)
    p   = (a0 + a1) * 0.5           # stays in original dtype

    # ── add in original dtype (matches PyTorch), then layer norm ─────────────
    xf  = (g + p).to(tl.float32)
    m   = tl.sum(xf, axis=0) / N
    d   = tl.where(mask, xf - m, 0.0)
    v   = tl.sum(d * d, axis=0) / N
    xn  = d * tl.rsqrt(v + eps)

    w   = tl.load(w_ptr    + offsets, mask=mask, other=1.0).to(tl.float32)
    bv  = tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = (xn * w + bv).to(cv.dtype)

    tl.store(out_ptr + batch_id * L * N + t * N + offsets, out, mask=mask)


@torch.fx.wrap
def fused_gelu_avgpool_add_trans_ln_dropout(conv_out, in_3, weight, bias_w):
    """
    Replaces:
        tmp_4  = F.gelu(conv_out)                                     # [B,N,125]
        tmp_5  = avg_pool1d(in_3, (2,), (2,), (0,), False, True)      # [B,N,124]
        tmp_6  = tmp_5[..., :124]   (no-op)
        tmp_7  = tmp_4[..., :124]   (slice 125→124)
        tmp_8  = tmp_6 + tmp_7
        tmp_9  = tmp_8.transpose(1, 2)
        tmp_10 = F.layer_norm(tmp_9, (768,), weight, bias, 1e-5)
        tmp_11 = F.dropout(tmp_10, 0.1, training=False)                # identity
    """
    batch, N, _ = in_3.shape      # [1, 768, 249]
    L = in_3.shape[2] // 2        # 124
    out = torch.empty((batch, L, N), dtype=in_3.dtype, device=in_3.device)

    _fused_gelu_avgpool_add_ln_kernel[(batch * L,)](
        conv_out, in_3, weight, bias_w, out,
        N, L, 1e-5,
        conv_out.stride(0), conv_out.stride(1),
        in_3.stride(0), in_3.stride(1),
    )
    return out


# ─── pattern / replacement API ───────────────────────────────────────────────

def pattern(in_3, conv1d, in_1, in_0):
    """
    Wide pattern: gelu + avg_pool1d + both slices + add + transpose
                  + layer_norm + dropout
    in_3   = original hidden states [B, N, 249]
    conv1d = raw conv1d output      [B, N, 125]  (gelu applied inside)
    in_1   = layer-norm weight
    in_0   = layer-norm bias
    """
    tmp_4  = torch.nn.functional.gelu(conv1d)
    tmp_5  = torch.avg_pool1d(in_3, (2,), (2,), (0,), False, True)
    tmp_6  = tmp_5[(Ellipsis, slice(None, 124, None))]
    tmp_7  = tmp_4[(Ellipsis, slice(None, 124, None))]
    tmp_8  = tmp_6 + tmp_7
    tmp_9  = tmp_8.transpose(1, 2)
    tmp_10 = torch.nn.functional.layer_norm(tmp_9, (768,), in_1, in_0, 1e-05)
    tmp_11 = torch.nn.functional.dropout(tmp_10, 0.1, False, False)
    return tmp_11


def replacement_args(in_3, conv1d, in_1, in_0):
    # conv_out=conv1d, in_3=in_3, weight=in_1, bias=in_0
    return (conv1d, in_3, in_1, in_0)


def replacement_func():
    return fused_gelu_avgpool_add_trans_ln_dropout