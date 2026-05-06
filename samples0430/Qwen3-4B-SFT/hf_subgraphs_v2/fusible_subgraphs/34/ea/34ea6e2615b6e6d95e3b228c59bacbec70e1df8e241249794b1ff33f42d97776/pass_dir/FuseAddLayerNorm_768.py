import torch
import triton
import triton.language as tl


# -----------------------------------------------------------------------
# Pattern: match add + layer_norm + flatten(2) + transpose(1,2)
# This fuses the residual add with the LN to eliminate intermediate buffers.
# Shapes: conv_out [1,768,16,16], y [1,768,16,16]
# -----------------------------------------------------------------------
def pattern(conv_out, y, weight, bias):
    added  = conv_out + y
    flat   = added.flatten(2)
    trans  = flat.transpose(1, 2)
    out    = torch.nn.functional.layer_norm(trans, (768,), weight, bias, 1e-05)
    return out


def replacement_args(conv_out, y, weight, bias):
    return (conv_out, y, weight, bias)


# -----------------------------------------------------------------------
# Triton kernel: fused residual-add + layer-norm
#   Reads conv_out [N,S,C] (non-contiguous, e.g. after transpose(1,2))
#   Reads y [N,S,C]      (contiguous)
#   Writes output [N,S,C] contiguous
# -----------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
    ],
    key=['S', 'C'],
)
@triton.jit
def add_layernorm_768_kernel(
    x_ptr,       # conv_out : [N, S, C]
    y_ptr,       # residual : [N, S, C]
    w_ptr,       # LN weight  [C]
    b_ptr,       # LN bias    [C]
    out_ptr,     # output     [N, S, C]
    stride_xn,   # stride of x along N
    stride_xs,   # stride of x along S (may be non-1 after transpose)
    stride_yc,   # stride of y along C (contiguous, = 1)
    stride_on,   # stride of out along N (contiguous = S*C)
    stride_os,   # stride of out along S (contiguous = C)
    S,           # total sequence length (= H*W)
    C: tl.constexpr,   # = 768
):
    row_idx = tl.program_id(0)          # one program per (n, s) position
    n = row_idx // S
    s = row_idx % S

    offs_c = tl.arange(0, C)            # no padding: C == BLOCK_SIZE

    # --- load conv_out[n,s,:] with physical strides ---
    x_base = n * stride_xn + s * stride_xs
    xv = tl.load(x_ptr + x_base + offs_c * stride_yc)   # y stride_c = 1

    # --- load y[n,s,:] contiguously ---
    yv = tl.load(y_ptr + n * S * C + s * C + offs_c)

    # --- residual add (FMA style: loaded → add → store) ---
    z = xv + yv
    z_f32 = z.to(tl.float32)

    # --- layer norm ---
    mean  = tl.sum(z_f32) * (1.0 / C)
    diff  = z_f32 - mean
    var   = tl.sum(diff * diff) * (1.0 / C)
    rstd  = tl.rsqrt(var + 1e-5)
    xn    = diff * rstd

    w = tl.load(w_ptr + offs_c).to(tl.float32)
    b = tl.load(b_ptr + offs_c).to(tl.float32)
    out = (xn * w + b).to(z.dtype)

    # --- store output contiguously ---
    tl.store(out_ptr + n * stride_on + s * stride_os + offs_c, out)


@torch.fx.wrap
def fused_add_layernorm_768(conv_out, y, weight, bias):
    """Fused residual-add + layer-norm for 768-channel model.
    Inputs:  x  [N, H, W, 768] (conv output, possibly non-contiguous after transpose)
             y  [N, H, W, 768] (residual, contiguous)
    Output:  [N, H*W, 768] contiguous
    """
    N = conv_out.shape[0]   # 1
    H = conv_out.shape[2]   # 16
    W = conv_out.shape[3]   # 16
    C = 768
    S = H * W               # 256

    out   = torch.empty((N, S, C), dtype=conv_out.dtype, device=conv_out.device)
    y_cn  = y.contiguous()

    add_strides_x  = conv_out.stride()           # [N,S,C] stride tuple
    add_strides_y  = y_cn.stride()               # contiguous: [N, S*C, C, 1]

    # Contiguous strides for output
    out_strides = (S * C, C, 1)

    add_ln_kernel[(N * S,)](
        conv_out, y_cn, weight, bias, out,
        add_strides_x[0],    # stride xn
        add_strides_x[1],    # stride xs (may != 1)
        add_strides_y[3],    # stride yc = 1 (contiguous last dim)
        out_strides[0],      # stride on
        out_strides[1],      # stride os
        S=S,
        C=C,
    )
    return out


def replacement_func():
    return fused_add_layernorm_768