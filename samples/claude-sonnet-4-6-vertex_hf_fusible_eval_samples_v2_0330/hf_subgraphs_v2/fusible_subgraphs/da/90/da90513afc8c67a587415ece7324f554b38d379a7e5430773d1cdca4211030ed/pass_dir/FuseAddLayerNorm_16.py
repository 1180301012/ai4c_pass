import torch
import triton
import triton.language as tl


@triton.jit
def _fused_add_ln_16_kernel(
    in2_ptr, in3_ptr, weight_ptr, bias_ptr, out_ptr,
    eps,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused add + layer_norm kernel for N=16.
    - N=16, BLOCK_SIZE=16 → single pass, no loop, no masking.
    - All 16 elements fit in registers.
    """
    row  = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)

    # Load inputs
    x2 = tl.load(in2_ptr + row * N + cols)
    x3 = tl.load(in3_ptr + row * N + cols)
    xf = x2.to(tl.float32) + x3.to(tl.float32)

    # Mean and variance in a single pass (data fits in registers)
    mean = tl.sum(xf, axis=0) / N
    xc   = xf - mean
    var  = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    # Affine transform
    w   = tl.load(weight_ptr + cols).to(tl.float32)
    b   = tl.load(bias_ptr   + cols).to(tl.float32)
    y   = xc * rstd * w + b

    tl.store(out_ptr + row * N + cols, y.to(x2.dtype))


@torch.fx.wrap
def _fused_add_ln_16(in_0, in_1, in_2, in_3):
    """in_0=bias, in_1=weight, in_2=first addend, in_3=second addend"""
    N          = 16
    BLOCK_SIZE = 16           # Exactly N, single-pass, no masking
    total_rows = in_2.numel() // N
    out        = torch.empty_like(in_2)

    _fused_add_ln_16_kernel[(total_rows,)](
        in_2, in_3, in_1, in_0, out,
        1e-05,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=1,
    )
    return out


def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_2 + in_3
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (16,), in_1, in_0, 1e-05)
    return tmp_3


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return _fused_add_ln_16