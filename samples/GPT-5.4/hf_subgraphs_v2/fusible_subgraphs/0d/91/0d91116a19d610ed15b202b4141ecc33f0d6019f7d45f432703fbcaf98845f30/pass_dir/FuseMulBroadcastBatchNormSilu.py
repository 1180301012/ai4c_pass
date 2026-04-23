import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    tmp_4 = in_5 * in_4
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 = torch.nn.functional.silu(tmp_5, inplace=True)
    return (tmp_6,)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


_PARAM_CACHE = {}


def _get_cached_param(src, device):
    if src.device == device:
        return src
    key = (src.data_ptr(), tuple(src.shape), str(src.dtype), device.type, device.index)
    cached = _PARAM_CACHE.get(key)
    if cached is None:
        cached = torch.empty(src.shape, dtype=src.dtype, device=device)
        cached.copy_(src)
        _PARAM_CACHE[key] = cached
    return cached


@triton.jit
def fused_mul_bn_silu_kernel(
    x_ptr,
    gate_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    C,
    H,
    W,
    sx0,
    sx1,
    sx2,
    sx3,
    sg0,
    sg1,
    sm0,
    sv0,
    sw0,
    sb0,
    so0,
    so1,
    so2,
    so3,
    BLOCK_HW: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_nc = tl.program_id(1)

    hw_offsets = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    hw = H * W
    mask = hw_offsets < hw

    n = pid_nc // C
    c = pid_nc % C

    mean = tl.load(mean_ptr + c * sm0).to(tl.float32)
    var = tl.load(var_ptr + c * sv0).to(tl.float32)
    gamma = tl.load(weight_ptr + c * sw0).to(tl.float32)
    beta = tl.load(bias_ptr + c * sb0).to(tl.float32)
    gate = tl.load(gate_ptr + n * sg0 + c * sg1).to(tl.float32)

    inv_std = tl.rsqrt(var + 1.0e-5)
    scale = gate * gamma * inv_std
    shift = beta - mean * gamma * inv_std

    h_idx = hw_offsets // W
    w_idx = hw_offsets % W

    x_ptrs = x_ptr + n * sx0 + c * sx1 + h_idx * sx2 + w_idx * sx3
    out_ptrs = out_ptr + n * so0 + c * so1 + h_idx * so2 + w_idx * so3

    x = tl.load(x_ptrs, mask=mask, other=0).to(tl.float32)
    y = x * scale + shift
    y = y * tl.sigmoid(y)
    tl.store(out_ptrs, y, mask=mask)



@torch.fx.wrap
def fused_mul_batchnorm_silu(mean, var, bias, weight, gate, x):
    mean = _get_cached_param(mean, x.device)
    var = _get_cached_param(var, x.device)
    bias = _get_cached_param(bias, x.device)
    weight = _get_cached_param(weight, x.device)

    out = torch.empty_like(x)

    n = x.shape[0]
    c = x.shape[1]
    h = x.shape[2]
    w = x.shape[3]

    if x.dtype == torch.float32:
        block_hw = 256
        num_warps = 4
    else:
        block_hw = 512
        num_warps = 8

    grid = (triton.cdiv(h * w, block_hw), n * c)

    fused_mul_bn_silu_kernel[grid](
        x,
        gate,
        mean,
        var,
        weight,
        bias,
        out,
        c,
        h,
        w,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        gate.stride(0),
        gate.stride(1),
        mean.stride(0),
        var.stride(0),
        weight.stride(0),
        bias.stride(0),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        BLOCK_HW=block_hw,
        num_warps=num_warps,
    )
    return (out,)



def replacement_func():
    return fused_mul_batchnorm_silu