import torch
import triton
import triton.language as tl


@triton.jit
def _fused_add_ln_1024_kernel(
    in2_ptr, in3_ptr, weight_ptr, bias_ptr, out_ptr,
    eps,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Single-pass fused add + layer_norm for N=1024.
    BLOCK_SIZE=1024=N exactly — no masking, no loop overhead.
    Loads x2,x3 once; keeps x in registers throughout.
    """
    row  = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    # N == BLOCK_SIZE == 1024, no masking needed
    x2 = tl.load(in2_ptr + row * N + cols)
    x3 = tl.load(in3_ptr + row * N + cols)
    xf = x2.to(tl.float32) + x3.to(tl.float32)

    # Stats from registers
    x_sum  = tl.sum(xf,      axis=0)
    x2_sum = tl.sum(xf * xf, axis=0)
    mean   = x_sum  / N
    var    = x2_sum / N - mean * mean
    rstd   = 1.0 / tl.sqrt(var + eps)

    # Affine + store
    w = tl.load(weight_ptr + cols).to(tl.float32)
    b = tl.load(bias_ptr   + cols).to(tl.float32)
    y = ((xf - mean) * rstd) * w + b
    tl.store(out_ptr + row * N + cols, y.to(x2.dtype))


@torch.fx.wrap
def _fused_add_ln_1024(in_0, in_1, in_2, in_3):
    """in_0=bias, in_1=weight, in_2=first addend, in_3=second addend"""
    N          = 1024
    BLOCK_SIZE = 1024     # exactly N, no masking
    total_rows = in_2.numel() // N
    out        = torch.empty_like(in_2)

    _fused_add_ln_1024_kernel[(total_rows,)](
        in_2, in_3, in_1, in_0, out,
        1e-05,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    return out


def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_2 + in_3
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (1024,), in_1, in_0, 1e-05)
    return tmp_3


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return _fused_add_ln_1024