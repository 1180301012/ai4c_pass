"""
Shared Triton kernels for fused operations.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _add_ln_fwd_kernel_16(
    x_ptr, y_ptr, out_ptr,
    weight_ptr, bias_ptr,
    eps: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused add + layer norm for H = 16."""
    row_idx = tl.program_id(0)
    cs = row_idx * D + tl.arange(0, BLOCK_D)

    x = tl.load(x_ptr + cs).to(tl.float32)
    y = tl.load(y_ptr + cs).to(tl.float32)
    z = x + y

    mean = tl.sum(z, axis=0) / D
    diff = z - mean
    var = tl.sum(diff * diff, axis=0) / D
    inv_std = 1.0 / tl.sqrt(var + eps)
    z_norm = diff * inv_std

    w = tl.load(weight_ptr + tl.arange(0, BLOCK_D)).to(tl.float32)
    b = tl.load(bias_ptr + tl.arange(0, BLOCK_D)).to(tl.float32)

    out = z_norm * w + b
    tl.store(out_ptr + cs, out.to(z.dtype))


@triton.jit
def _add_ln_fwd_kernel_256(
    x_ptr, y_ptr, out_ptr,
    weight_ptr, bias_ptr,
    eps: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused add + layer norm for H = 256."""
    row_idx = tl.program_id(0)
    cs = row_idx * D + tl.arange(0, BLOCK_D)

    x = tl.load(x_ptr + cs).to(tl.float32)
    y = tl.load(y_ptr + cs).to(tl.float32)
    z = x + y

    mean = tl.sum(z, axis=0) / D
    diff = z - mean
    var = tl.sum(diff * diff, axis=0) / D
    inv_std = 1.0 / tl.sqrt(var + eps)
    z_norm = diff * inv_std

    w = tl.load(weight_ptr + tl.arange(0, BLOCK_D)).to(tl.float32)
    b = tl.load(bias_ptr + tl.arange(0, BLOCK_D)).to(tl.float32)

    out = z_norm * w + b
    tl.store(out_ptr + cs, out.to(z.dtype))


@triton.jit
def _add_ln_fwd_kernel_768(
    x_ptr, y_ptr, out_ptr,
    weight_ptr, bias_ptr,
    eps: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused add + layer norm for H = 768."""
    row_idx = tl.program_id(0)
    cs = row_idx * D + tl.arange(0, BLOCK_D)

    x = tl.load(x_ptr + cs).to(tl.float32)
    y = tl.load(y_ptr + cs).to(tl.float32)
    z = x + y

    mean = tl.sum(z, axis=0) / D
    diff = z - mean
    var = tl.sum(diff * diff, axis=0) / D
    inv_std = 1.0 / tl.sqrt(var + eps)
    z_norm = diff * inv_std

    w = tl.load(weight_ptr + tl.arange(0, BLOCK_D)).to(tl.float32)
    b = tl.load(bias_ptr + tl.arange(0, BLOCK_D)).to(tl.float32)

    out = z_norm * w + b
    tl.store(out_ptr + cs, out.to(z.dtype))


@triton.jit
def _add_ln_fwd_kernel_1024(
    x_ptr, y_ptr, out_ptr,
    weight_ptr, bias_ptr,
    eps: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused add + layer norm for H = 1024."""
    row_idx = tl.program_id(0)
    cs = row_idx * D + tl.arange(0, BLOCK_D)

    x = tl.load(x_ptr + cs).to(tl.float32)
    y = tl.load(y_ptr + cs).to(tl.float32)
    z = x + y

    mean = tl.sum(z, axis=0) / D
    diff = z - mean
    var = tl.sum(diff * diff, axis=0) / D
    inv_std = 1.0 / tl.sqrt(var + eps)
    z_norm = diff * inv_std

    w = tl.load(weight_ptr + tl.arange(0, BLOCK_D)).to(tl.float32)
    b = tl.load(bias_ptr + tl.arange(0, BLOCK_D)).to(tl.float32)

    out = z_norm * w + b
    tl.store(out_ptr + cs, out.to(z.dtype))


@triton.jit
def _mask_fill_kernel(
    in_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel: (1 - x.to(float32)).to(bool) ? -3.4e38 : 0.0"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    in_vals = tl.load(in_ptr + offsets, mask=mask, other=1)
    vals_f32 = in_vals.to(tl.float32)
    one_f32 = tl.full_like(vals_f32, 1.0)
    diff = one_f32 - vals_f32  # 1 - int_value
    is_zero = diff.to(tl.int1)  # 1 where diff==0 (int_val==1), 0 else
    result = tl.where(is_zero == 1, -3.4028234663852886e+38, 0.0)
    tl.store(out_ptr + offsets, result, mask=mask)


def _launch_add_ln_kernel(x, y, weight, bias, H, BLOCK_D, EPS):
    EPS_VAL = float(EPS)
    D = H
    out = torch.empty_like(x)
    num_rows = x.numel() // D
    grid = (num_rows,)
    if H == 16:
        _add_ln_fwd_kernel_16[grid](
            x, y, out, weight, bias,
            eps=EPS_VAL, D=D, BLOCK_D=BLOCK_D,
        )
    elif H == 256:
        _add_ln_fwd_kernel_256[grid](
            x, y, out, weight, bias,
            eps=EPS_VAL, D=D, BLOCK_D=BLOCK_D,
        )
    elif H == 768:
        _add_ln_fwd_kernel_768[grid](
            x, y, out, weight, bias,
            eps=EPS_VAL, D=D, BLOCK_D=BLOCK_D,
        )
    else:
        _add_ln_fwd_kernel_1024[grid](
            x, y, out, weight, bias,
            eps=EPS_VAL, D=D, BLOCK_D=BLOCK_D,
        )
    return out


def _placeholder_16(x, y, weight, bias):
    # placeholder branch (never executed for this route)
    return _launch_add_ln_kernel(x, y, weight, bias, 16, 16, 1e-5)

def _placeholder_256(x, y, weight, bias):
    return _launch_add_ln_kernel(x, y, weight, bias, 256, 256, 1e-5)

def _placeholder_768(x, y, weight, bias):
    return _launch_add_ln_kernel(x, y, weight, bias, 768, 256, 1e-5)

def _placeholder_1024(x, y, weight, bias):
    return _launch_add_ln_kernel(x, y, weight, bias, 1024, 1024, 1e-5)



@torch.fx.wrap
def shared_dispatch_fused_add_ln(x, y, weight, bias, route):
    if route == "_16":
        return _launch_add_ln_kernel(x, y, weight, bias, 16, 16, 1e-5)
    elif route == "_256":
        return _launch_add_ln_kernel(x, y, weight, bias, 256, 256, 1e-5)
    elif route == "_768":
        return _launch_add_ln_kernel(x, y, weight, bias, 768, 256, 1e-5)
    else:  # _1024 or "mask"
        if route == "_1024":
            return _launch_add_ln_kernel(x, y, weight, bias, 1024, 1024, 1e-5)
        else:  # "mask" — x=in_5, a1/a2/a3 are dummy (unused)
            n_elements = x.numel()
            out = torch.empty_like(x)
            BLOCK_SIZE = 256
            n_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
            _mask_fill_kernel[(n_blocks,)](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
            return out


# Alias so FuseMaskFill uses the same replacement_func object
fused_mask_fill = shared_dispatch_fused_add_ln