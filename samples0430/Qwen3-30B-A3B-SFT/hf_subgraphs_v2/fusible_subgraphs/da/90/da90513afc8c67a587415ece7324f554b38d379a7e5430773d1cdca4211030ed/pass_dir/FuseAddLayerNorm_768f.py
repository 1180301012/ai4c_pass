import torch
import triton
import triton.language as tl
from pass_dir.shared_add_ln import dispatch_fused_add_ln  # shared object across all passes





@triton.jit
def _fused_add_ln_768_kernel(
    x1_ptr, x2_ptr, w_ptr, b_ptr, out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    row_offset = row * N

    x1 = tl.load(x1_ptr + row_offset + offsets, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x2_ptr + row_offset + offsets, mask=mask, other=0.0).to(tl.float32)
    z = x1 + x2

    mean = tl.sum(z, axis=0) / N
    diff = tl.where(mask, z - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = tl.math.rsqrt(var + 1e-05)
    z_norm = diff * rstd

    w = tl.load(w_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = z_norm * w + b
    tl.store(out_ptr + row_offset + offsets, out, mask=mask)


@triton.jit
def _fused_add_ln_1024_kernel(
    x1_ptr, x2_ptr, w_ptr, b_ptr, out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    row_offset = row * N

    x1 = tl.load(x1_ptr + row_offset + offsets, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x2_ptr + row_offset + offsets, mask=mask, other=0.0).to(tl.float32)
    z = x1 + x2

    mean = tl.sum(z, axis=0) / N
    diff = tl.where(mask, z - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = tl.math.rsqrt(var + 1e-05)
    z_norm = diff * rstd

    w = tl.load(w_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = z_norm * w + b
    tl.store(out_ptr + row_offset + offsets, out, mask=mask)


@triton.jit
def _fused_add_ln_16_kernel(
    x1_ptr, x2_ptr, w_ptr, b_ptr, out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    row_offset = row * N

    x1 = tl.load(x1_ptr + row_offset + offsets, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x2_ptr + row_offset + offsets, mask=mask, other=0.0).to(tl.float32)
    z = x1 + x2

    mean = tl.sum(z, axis=0) / N
    diff = tl.where(mask, z - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = tl.math.rsqrt(var + 1e-05)
    z_norm = diff * rstd

    w = tl.load(w_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = z_norm * w + b
    tl.store(out_ptr + row_offset + offsets, out, mask=mask)


def _run_768(bias, weight, x1, x2):
    N = 768
    total_rows = x1.numel() // N
    out = torch.empty_like(x1)
    _fused_add_ln_768_kernel[(total_rows,)](
        x1, x2, weight, bias, out,
        N=N, BLOCK_SIZE=1024, num_warps=8,
    )
    return out


def _run_1024(bias, weight, x1, x2):
    N = 1024
    total_rows = x1.numel() // N
    out = torch.empty_like(x1)
    _fused_add_ln_1024_kernel[(total_rows,)](
        x1, x2, weight, bias, out,
        N=N, BLOCK_SIZE=1024, num_warps=8,
    )
    return out


def _run_16(bias, weight, x1, x2):
    N = 16
    total_rows = x1.numel() // N
    out = torch.empty_like(x1)
    _fused_add_ln_16_kernel[(total_rows,)](
        x1, x2, weight, bias, out,
        N=N, BLOCK_SIZE=16, num_warps=4,
    )
    return out


@torch.fx.wrap
def _dispatch_fused_add_ln(bias, weight, x1, x2, route):
    if route == "768":
        return _run_768(bias, weight, x1, x2)
    elif route == "1024":
        return _run_1024(bias, weight, x1, x2)
    elif route == "16":
        return _run_16(bias, weight, x1, x2)
    else:
        return _run_768(bias, weight, x1, x2)


def pattern(in_0, in_1, in_2, in_3):
    # Reversed: in_3 + in_2
    tmp_2 = in_3 + in_2
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (768,), in_1, in_0, 1e-05)
    return tmp_3


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3, "768")


def replacement_func():
    return dispatch_fused_add_ln