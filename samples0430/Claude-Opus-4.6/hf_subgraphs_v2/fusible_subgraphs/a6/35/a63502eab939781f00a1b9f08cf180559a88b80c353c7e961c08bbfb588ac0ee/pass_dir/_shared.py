import torch
import triton
import triton.language as tl


@triton.jit
def _fused_roll_ln_add_kernel_768(
    in3_ptr, weight_ptr, bias_ptr, residual_ptr, out_ptr,
    s1, s2, s3, s4,
    BLOCK_C: tl.constexpr,
):
    row_idx = tl.program_id(0)
    H: tl.constexpr = 32
    W: tl.constexpr = 32
    C: tl.constexpr = 768
    ws: tl.constexpr = 8
    shift: tl.constexpr = 4

    i = row_idx // W
    j = row_idx % W
    src_i = (i - shift + H) % H
    src_j = (j - shift + W) % W

    a = src_i // ws
    b = src_i % ws
    c = src_j // ws
    d = src_j % ws

    base = a * s1 + b * s2 + c * s3 + d * s4

    col_offsets = tl.arange(0, BLOCK_C)
    mask = col_offsets < C

    x = tl.load(in3_ptr + base + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Compute mean
    mean = tl.sum(x, axis=0) / C

    # Compute variance with correction for masked positions
    x_centered = x - mean
    var_raw = tl.sum(x_centered * x_centered, axis=0) / C
    var = var_raw - (BLOCK_C - C) * mean * mean / C

    inv_std = tl.math.rsqrt(var + 1e-5)
    x_norm = x_centered * inv_std

    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    x_ln = x_norm * weight + bias

    residual = tl.load(residual_ptr + row_idx * C + col_offsets, mask=mask, other=0.0).to(tl.float32)
    output = residual + x_ln

    tl.store(out_ptr + row_idx * C + col_offsets, output, mask=mask)


@triton.jit
def _fused_roll_ln_add_kernel_384(
    in3_ptr, weight_ptr, bias_ptr, residual_ptr, out_ptr,
    s1, s2, s3, s4,
    BLOCK_C: tl.constexpr,
):
    row_idx = tl.program_id(0)
    H: tl.constexpr = 64
    W: tl.constexpr = 64
    C: tl.constexpr = 384
    ws: tl.constexpr = 8
    shift: tl.constexpr = 4

    i = row_idx // W
    j = row_idx % W
    src_i = (i - shift + H) % H
    src_j = (j - shift + W) % W

    a = src_i // ws
    b = src_i % ws
    c = src_j // ws
    d = src_j % ws

    base = a * s1 + b * s2 + c * s3 + d * s4

    col_offsets = tl.arange(0, BLOCK_C)
    mask = col_offsets < C

    x = tl.load(in3_ptr + base + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Compute mean
    mean = tl.sum(x, axis=0) / C

    # Compute variance with correction for masked positions
    x_centered = x - mean
    var_raw = tl.sum(x_centered * x_centered, axis=0) / C
    var = var_raw - (BLOCK_C - C) * mean * mean / C

    inv_std = tl.math.rsqrt(var + 1e-5)
    x_norm = x_centered * inv_std

    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    x_ln = x_norm * weight + bias

    residual = tl.load(residual_ptr + row_idx * C + col_offsets, mask=mask, other=0.0).to(tl.float32)
    output = residual + x_ln

    tl.store(out_ptr + row_idx * C + col_offsets, output, mask=mask)


def _run_32_768(in_0, in_1, in_2, in_3):
    H, W, C = 32, 32, 768
    N = H * W
    out = torch.empty(1, N, C, dtype=in_2.dtype, device=in_2.device)
    s = in_3.stride()
    if len(s) == 6:
        _fused_roll_ln_add_kernel_768[(N,)](
            in_3, in_1, in_0, in_2, out,
            s[1], s[2], s[3], s[4],
            BLOCK_C=1024, num_warps=2, num_stages=1,
        )
    else:
        x = in_3.contiguous()
        _fused_roll_ln_add_kernel_768[(N,)](
            x, in_1, in_0, in_2, out,
            8 * 4 * 8 * 768, 4 * 8 * 768, 8 * 768, 768,
            BLOCK_C=1024, num_warps=2, num_stages=1,
        )
    return out


def _run_64_384(in_0, in_1, in_2, in_3):
    H, W, C = 64, 64, 384
    N = H * W
    out = torch.empty(1, N, C, dtype=in_2.dtype, device=in_2.device)
    s = in_3.stride()
    if len(s) == 6:
        _fused_roll_ln_add_kernel_384[(N,)](
            in_3, in_1, in_0, in_2, out,
            s[1], s[2], s[3], s[4],
            BLOCK_C=512, num_warps=8, num_stages=2,
        )
    else:
        x = in_3.contiguous()
        _fused_roll_ln_add_kernel_384[(N,)](
            x, in_1, in_0, in_2, out,
            8 * 8 * 8 * 384, 8 * 8 * 384, 8 * 384, 384,
            BLOCK_C=512, num_warps=8, num_stages=2,
        )
    return out


@torch.fx.wrap
def dispatch_fused_roll_ln_add(in_0, in_1, in_2, in_3, route):
    if route == "route_32_768":
        return _run_32_768(in_0, in_1, in_2, in_3)
    elif route == "route_64_384":
        return _run_64_384(in_0, in_1, in_2, in_3)


def replacement_func():
    return dispatch_fused_roll_ln_add