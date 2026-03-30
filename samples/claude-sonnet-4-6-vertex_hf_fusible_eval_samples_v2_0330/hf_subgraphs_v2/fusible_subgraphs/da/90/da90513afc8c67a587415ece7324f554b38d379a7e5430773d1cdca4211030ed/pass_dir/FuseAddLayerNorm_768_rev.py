import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16),
    ],
    key=['N'],
)
@triton.jit
def _fused_add_ln_768r_kernel(
    in2_ptr, in3_ptr, weight_ptr, bias_ptr, out_ptr,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # Load inputs
    x2 = tl.load(in2_ptr + row * N + cols, mask=mask, other=0.0)
    x3 = tl.load(in3_ptr + row * N + cols, mask=mask, other=0.0)

    # Add (commutative, order doesn't matter for correctness)
    xf = x2.to(tl.float32) + x3.to(tl.float32)

    # Zero out padding for correct reduction
    xf_valid = tl.where(mask, xf, 0.0)

    # Mean
    mean = tl.sum(xf_valid, axis=0) / N

    # Variance
    xf_cent = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(xf_cent * xf_cent, axis=0) / N

    # Normalize
    rstd = 1.0 / tl.sqrt(var + eps)
    x_norm = xf_cent * rstd

    # Affine transform
    w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr  + cols, mask=mask, other=0.0).to(tl.float32)
    out = x_norm * w + b

    # Store back in original dtype
    tl.store(out_ptr + row * N + cols, out.to(x2.dtype), mask=mask)


@torch.fx.wrap
def _fused_add_ln_768r(in_0, in_1, in_2, in_3):
    """in_0=bias, in_1=weight; pattern: in_3 + in_2 (reversed)"""
    N = 768
    total_rows = in_2.numel() // N
    out = torch.empty_like(in_2)

    _fused_add_ln_768r_kernel[(total_rows,)](
        in_3, in_2, in_1, in_0, out,
        N, 1e-05,
    )
    return out


def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_3 + in_2
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (768,), in_1, in_0, 1e-05)
    return tmp_3


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return _fused_add_ln_768r