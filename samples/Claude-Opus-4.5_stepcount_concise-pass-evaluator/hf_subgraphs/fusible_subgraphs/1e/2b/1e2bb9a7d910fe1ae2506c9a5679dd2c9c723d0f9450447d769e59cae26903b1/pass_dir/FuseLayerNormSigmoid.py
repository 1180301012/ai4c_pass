import torch
import triton
import triton.language as tl


def pattern(bias, weight, x):
    """
    Match layer_norm + sigmoid pattern
    """
    tmp_2 = torch.nn.functional.layer_norm(x, (256,), weight, bias, 1e-05)
    tmp_4 = tmp_2.sigmoid()
    return tmp_4


def replacement_args(bias, weight, x):
    return (bias, weight, x)


@triton.jit
def fused_ln_sigmoid(
    X, B, W, O,
    stride,
    N: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, N)
    base = row * stride
    
    # Load data
    x = tl.load(X + base + cols).to(tl.float32)
    w = tl.load(W + cols).to(tl.float32)
    b = tl.load(B + cols).to(tl.float32)
    
    # LayerNorm + Sigmoid fused
    mean = tl.sum(x, 0) / N
    xc = x - mean
    var = tl.sum(xc * xc, 0) / N
    y = tl.sigmoid(xc * tl.rsqrt(var + EPS) * w + b)
    
    tl.store(O + base + cols, y)


@torch.fx.wrap
def kernel_wrapper(bias, weight, x):
    out = torch.empty_like(x)
    fused_ln_sigmoid[(x.numel() // 256,)](
        x, bias, weight, out, 256,
        N=256, EPS=1e-5, num_warps=8
    )
    return out


def replacement_func():
    return kernel_wrapper