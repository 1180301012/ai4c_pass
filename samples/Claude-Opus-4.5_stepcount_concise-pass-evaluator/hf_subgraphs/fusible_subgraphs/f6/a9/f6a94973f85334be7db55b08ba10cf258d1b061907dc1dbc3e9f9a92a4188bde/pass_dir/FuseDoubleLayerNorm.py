import torch
import triton
import triton.language as tl


def pattern(x, normalized_shape1, weight1, bias1, normalized_shape2, weight2, bias2):
    ln1 = torch.nn.functional.layer_norm(x, normalized_shape1, weight1, bias1, 1e-05)
    ln2 = torch.nn.functional.layer_norm(ln1, normalized_shape2, weight2, bias2, 1e-05)
    return (ln1, ln2)


def replacement_args(x, normalized_shape1, weight1, bias1, normalized_shape2, weight2, bias2):
    return (x, weight1, bias1, weight2, bias2)


@triton.jit
def fused_ln_kernel(
    x_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, out1_ptr, out2_ptr,
    hidden_dim, eps: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < hidden_dim
    base = row * hidden_dim
    
    # Load once
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    w1 = tl.load(w1_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    b1 = tl.load(b1_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    w2 = tl.load(w2_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    b2 = tl.load(b2_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    
    inv_n = 1.0 / hidden_dim
    
    # First LayerNorm
    m1 = tl.sum(x, 0) * inv_n
    d1 = tl.where(mask, x - m1, 0.0)
    v1 = tl.sum(d1 * d1, 0) * inv_n
    y1 = d1 * tl.rsqrt(v1 + eps) * w1 + b1
    tl.store(out1_ptr + base + offs, y1, mask=mask)
    
    # Second LayerNorm (reuse registers)
    y1_m = tl.where(mask, y1, 0.0)
    m2 = tl.sum(y1_m, 0) * inv_n
    d2 = tl.where(mask, y1 - m2, 0.0)
    v2 = tl.sum(d2 * d2, 0) * inv_n
    y2 = d2 * tl.rsqrt(v2 + eps) * w2 + b2
    tl.store(out2_ptr + base + offs, y2, mask=mask)


@torch.fx.wrap
def fused_double_layer_norm_impl(x, weight1, bias1, weight2, bias2):
    orig_shape = x.shape
    hidden_dim = orig_shape[-1]
    x_2d = x.view(-1, hidden_dim)
    n_rows = x_2d.shape[0]
    
    out1 = torch.empty_like(x_2d)
    out2 = torch.empty_like(x_2d)
    
    # Optimized configurations
    if hidden_dim <= 1024:
        BLOCK = 1024
        nwarps = 4
    else:
        BLOCK = 2048
        nwarps = 8
    
    fused_ln_kernel[(n_rows,)](
        x_2d, weight1, bias1, weight2, bias2,
        out1, out2, hidden_dim,
        eps=1e-05, BLOCK=BLOCK,
        num_warps=nwarps,
        num_stages=2,
    )
    
    return out1.view(orig_shape), out2.view(orig_shape)


def fused_double_layer_norm(x, weight1, bias1, weight2, bias2):
    r = fused_double_layer_norm_impl(x, weight1, bias1, weight2, bias2)
    return (r[0], r[1])


def replacement_func():
    return fused_double_layer_norm