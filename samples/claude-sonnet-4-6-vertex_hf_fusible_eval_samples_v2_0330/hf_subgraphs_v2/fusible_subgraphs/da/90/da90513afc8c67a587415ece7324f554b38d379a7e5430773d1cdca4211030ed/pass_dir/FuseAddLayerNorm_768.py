import torch
import triton
import triton.language as tl


@triton.jit
def _fused_add_ln_768_kernel(
    in2_ptr, in3_ptr, weight_ptr, bias_ptr, out_ptr,
    eps,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Single-pass fused add + layer_norm.
    - N, BLOCK_SIZE constexpr → static mask, compiler optimizations.
    - Reads x2, x3 ONCE; keeps x in registers for stats + normalization.
    - HBM: 2 reads (x2,x3) + small (weight,bias in L2) + 1 write (out).
    """
    row  = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # Single global read
    x2 = tl.load(in2_ptr + row * N + cols, mask=mask, other=0.0)
    x3 = tl.load(in3_ptr + row * N + cols, mask=mask, other=0.0)
    xf = x2.to(tl.float32) + x3.to(tl.float32)
    xv = tl.where(mask, xf, 0.0)

    # Stats from registers (no re-read)
    x_sum  = tl.sum(xv,      axis=0)
    x2_sum = tl.sum(xv * xv, axis=0)
    mean   = x_sum  / N
    var    = x2_sum / N - mean * mean
    rstd   = 1.0 / tl.sqrt(var + eps)

    # Affine transform + store (xf still in registers)
    w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr   + cols, mask=mask, other=0.0).to(tl.float32)
    y = ((xf - mean) * rstd) * w + b
    tl.store(out_ptr + row * N + cols, y.to(x2.dtype), mask=mask)


@torch.fx.wrap
def _fused_add_ln_768(in_0, in_1, in_2, in_3):
    """in_0=bias, in_1=weight, in_2=first addend, in_3=second addend"""
    N          = 768
    BLOCK_SIZE = 1024     # next power-of-2 >= N; mask handles 256 extras
    total_rows = in_2.numel() // N
    out        = torch.empty_like(in_2)

    _fused_add_ln_768_kernel[(total_rows,)](
        in_2, in_3, in_1, in_0, out,
        1e-05,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=2,
    )
    return out


def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_2 + in_3
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (768,), in_1, in_0, 1e-05)
    return tmp_3


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return _fused_add_ln_768