import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = torch.nn.functional.gelu(in_2, approximate='none')
    tmp_3 = tmp_2.flatten(2)
    tmp_4 = tmp_3.transpose(1, 2)
    tmp_5 = tmp_4.contiguous()
    tmp_6 = in_3 + tmp_5
    tmp_7 = tmp_6.permute(0, 2, 1)
    tmp_8 = tmp_7.view(1, 256, 8, 6)
    tmp_9 = tmp_8.view(1, 256, -1)
    tmp_10 = tmp_9.permute(0, 2, 1)
    tmp_11 = torch.nn.functional.layer_norm(tmp_10, (256,), in_1, in_0, 1e-06)
    tmp_12 = tmp_11.view(1, 8, 6, 256)
    return (tmp_10, tmp_12)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3, 'route_256_8_6')


@triton.jit
def _fused_pose_kernel(
    bias_ptr,
    weight_ptr,
    x_ptr,
    residual_ptr,
    out10_ptr,
    out12_ptr,
    B,
    M,
    H,
    W,
    C,
    x_s0,
    x_s1,
    x_s2,
    x_s3,
    r_s0,
    r_s1,
    r_s2,
    o10_s0,
    o10_s1,
    o10_s2,
    o12_s0,
    o12_s1,
    o12_s2,
    o12_s3,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // M
    row = pid - b * M
    offs_c = tl.arange(0, BLOCK_C)
    mask = offs_c < C

    h = row // W
    w = row - h * W

    x_ptrs = x_ptr + b * x_s0 + offs_c * x_s1 + h * x_s2 + w * x_s3
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    gelu = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))

    r_ptrs = residual_ptr + b * r_s0 + row * r_s1 + offs_c * r_s2
    r = tl.load(r_ptrs, mask=mask, other=0.0).to(tl.float32)

    v = gelu + r

    out10_ptrs = out10_ptr + b * o10_s0 + row * o10_s1 + offs_c * o10_s2
    tl.store(out10_ptrs, v, mask=mask)

    c_f32 = tl.cast(C, tl.float32)
    mean = tl.sum(v, axis=0) / c_f32
    centered = v - mean
    var = tl.sum(centered * centered, axis=0) / c_f32
    inv_std = tl.rsqrt(var + 1e-6)

    gamma = tl.load(weight_ptr + offs_c, mask=mask, other=1.0).to(tl.float32)
    beta = tl.load(bias_ptr + offs_c, mask=mask, other=0.0).to(tl.float32)
    y = centered * inv_std
    y = y * gamma + beta

    out12_ptrs = out12_ptr + b * o12_s0 + h * o12_s1 + w * o12_s2 + offs_c * o12_s3
    tl.store(out12_ptrs, y, mask=mask)


def _launch_fused(bias, weight, x, residual, h, w, c, block_c, num_warps):
    bsz = residual.shape[0]
    m = h * w
    out10 = torch.empty((bsz, m, c), device=residual.device, dtype=residual.dtype)
    out12 = torch.empty((bsz, h, w, c), device=residual.device, dtype=residual.dtype)
    grid = (bsz * m,)
    _fused_pose_kernel[grid](
        bias,
        weight,
        x,
        residual,
        out10,
        out12,
        bsz,
        m,
        h,
        w,
        c,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        residual.stride(0),
        residual.stride(1),
        residual.stride(2),
        out10.stride(0),
        out10.stride(1),
        out10.stride(2),
        out12.stride(0),
        out12.stride(1),
        out12.stride(2),
        out12.stride(3),
        BLOCK_C=block_c,
        num_warps=num_warps,
        num_stages=1,
    )
    return out10, out12


@torch.fx.wrap
def fused_pose_dispatch(bias, weight, x, residual, route):
    if route == 'route_32_64_48':
        return _launch_fused(bias, weight, x, residual, 64, 48, 32, 32, 1)
    if route == 'route_128_16_12':
        return _launch_fused(bias, weight, x, residual, 16, 12, 128, 128, 4)
    if route == 'route_256_8_6':
        return _launch_fused(bias, weight, x, residual, 8, 6, 256, 256, 8)
    raise RuntimeError('unknown route')


def replacement_func():
    return fused_pose_dispatch