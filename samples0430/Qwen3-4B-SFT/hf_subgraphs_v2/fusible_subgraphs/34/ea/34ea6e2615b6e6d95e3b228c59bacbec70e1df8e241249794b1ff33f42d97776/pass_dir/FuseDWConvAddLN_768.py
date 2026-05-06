"""
Full fusion pass: depthwise_conv2d + add + flatten + transpose + layer_norm
All three operations (conv, add, layer_norm) in ONE Triton kernel.
Pattern inputs:  weight[C,1,K,K], bias[C], residual[C,H,W]
Pattern output:  [H*W, C] contiguous (after flatten+transpose+LN)

Memory savings vs original 3-op pipeline:
  - No intermediate [C,H,W] conv output buffer
  - No intermediate [C,H,W] add result buffer
  - Direct layer-norm write to final [S,C] layout
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern
# ---------------------------------------------------------------------------
def pattern(weight, bias, residual, ln_w, ln_b):
    """Match depthwise-conv + residual-add + flatten + transpose + LN."""
    conv   = torch.conv2d(residual, weight, bias, (1, 1), (1, 1), (1, 1), 768)
    added  = conv + residual
    flat   = added.flatten(2)
    trans  = flat.transpose(1, 2)
    out    = torch.nn.functional.layer_norm(trans, (768,), ln_w, ln_b, 1e-05)
    return out


def replacement_args(weight, bias, residual, ln_w, ln_b):
    return (weight, bias, residual, ln_w, ln_b)


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------
@triton.jit
def _dw_conv_add_ln_768_kernel(
    x_ptr,           # residual  [1, C, H, W] contiguous
    w_ptr,           # weight    [C, 1, K, K] contiguous
    b_ptr,           # bias      [C]           contiguous
    beta_ptr,        # LN weight [C]           contiguous
    gamma_ptr,       # LN bias   [C]           contiguous
    out_ptr,         # output    [S, C] contiguous  (S = H*W)
    H:  tl.constexpr,       # spatial height
    W:  tl.constexpr,       # spatial width
    K:  tl.constexpr,       # kernel size (3 for 3x3)
    C:  tl.constexpr,       # channel count (768)
    W_OUT: tl.constexpr,    # W*C  (output stride for row = H*W*C)
    BLOCK_S: tl.constexpr,  # linear tile width (32 for 16-wide spatial)
    BLOCK_C: tl.constexpr,  # channel block size (= C for 1 channel/program,
                            # but we use whole C in one program)
):
    """Each program handles one (n=0, c) pair, all H*W spatial positions."""
    c  = tl.program_id(0)
    s  = tl.arange(0, BLOCK_S)          # 0 .. BLOCK_S-1

    # --- output row indices for this tile ---
    oh = s // W                           # which row  (0 .. OH-1)
    ow = s  % W                           # which col  (0 .. W-1)

    offs_c = tl.arange(0, BLOCK_C)
    c_msk  = offs_c < C

    # Weight [C, 1, K, K] treated as [C, KH*KW]
    WMASK  = tl.zeros([BLOCK_C], dtype=tl.int1) + (1 << (K - 1))   # bit-mask for K mask bits

    # Depthwise conv (3×3, pad=1, stride=1) — K loop fully unrolled at compile time
    acc = tl.zeros([BLOCK_S], dtype=tl.float32)

    for kh in range(K):
        for kw in range(K):
            ih = oh + (kh - 1)    # may be -1 (top pad) or >= H (bottom pad)
            iw = ow + (kw - 1)    # may be -1 (left pad) or >= W (right pad)

            h_ok = (ih >= 0) & (ih < H)   # [BLOCK_S]
            w_ok = (iw >= 0) & (iw < W)   # [BLOCK_S]
            mvt  = h_ok & w_ok             # vectorised mask

            wkv  = tl.load(w_ptr + c * K * K + kh * K + kw)
            xtv  = tl.load(x_ptr + c * H * W + ih * W + iw,
                           mask=mvt, other=0.0)
            acc  = acc + wkv * xtv.to(tl.float32)

    # Add residual
    rex = tl.load(x_ptr + c * H * W + oh * W + ow)
    acc = acc + rex.to(tl.float32)

    # layer norm  [BLOCK_S]
    acc_f = acc.to(tl.float32)
    acc_m = acc_f * c_msk.to(tl.float32)
    mean  = tl.sum(acc_m) * (1.0 / C)
    d     = acc_f - mean
    var   = tl.sum(d * d * c_msk.to(tl.float32)) * (1.0 / C)
    xn    = d * tl.rsqrt(var + 1e-5)

    wt = tl.load(beta_ptr + offs_c, mask=c_msk, other=1.0).to(tl.float32)
    bt = tl.load(gamma_ptr + offs_c, mask=c_msk, other=0.0).to(tl.float32)
    out_v = (xn * wt + bt).to(rex.dtype)

    # Write to [S, C] contiguous output: out[oh*W+ow, c]
    tl.store(out_ptr + (oh * W + ow) * C + offs_c, out_v, mask=c_msk)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fuse_dw_conv_add_ln_768(weight, bias, residual, ln_w, ln_b):
    """
    Fused depthwise-conv + add + flatten + transpose + layer-norm.
    Inputs:
      weight    [C, 1, K, K]
      bias      [C]
      residual  [1, C, H, W]
      ln_w, ln_b [C]
    Output:
      [H*W, C] contiguous (matches layer_norm output after .transpose(0,1))
    """
    C  = ln_w.shape[0]          # 768
    H  = weight.shape[2]        # 16
    W  = weight.shape[3]        # 16
    S  = H * W                  # 256

    out  = torch.empty((S, C), dtype=residual.dtype, device=residual.device)
    BS   = triton.next_power_of_2(max(W, C))   # 1024

    _dw_conv_add_ln_768_kernel[(C,)](
        residual, weight, bias, ln_w, ln_b, out,
        H=H, W=W, K=3, C=C, W_OUT=S, BLOCK_S=BS, BLOCK_C=BS,
        num_warps=4,
    )
    return out


def replacement_func():
    return fuse_dw_conv_add_ln_768