import torch
import triton
import triton.language as tl


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
def mul_broadcast_kernel(
    x_ptr,
    gate_ptr,
    out_ptr,
    n_elements,
    C,
    HW,
    sg0,
    sg1,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    nc = offsets // HW
    n = nc // C
    c = nc % C

    x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float32)
    gate = tl.load(gate_ptr + n * sg0 + c * sg1, mask=mask, other=0).to(tl.float32)
    y = x * gate
    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def bn_silu_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    C,
    HW,
    sm0,
    sv0,
    sw0,
    sb0,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    nc = offsets // HW
    c = nc % C

    mean = tl.load(mean_ptr + c * sm0, mask=mask, other=0).to(tl.float32)
    var = tl.load(var_ptr + c * sv0, mask=mask, other=0).to(tl.float32)
    gamma = tl.load(weight_ptr + c * sw0, mask=mask, other=0).to(tl.float32)
    beta = tl.load(bias_ptr + c * sb0, mask=mask, other=0).to(tl.float32)

    inv_std = tl.rsqrt(var + 1.0e-5)
    scale = gamma * inv_std
    shift = beta - mean * scale

    x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float32)
    y = x * scale + shift
    y = y * tl.sigmoid(y)
    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def mul_bn_kernel(
    x_ptr,
    gate_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    C,
    HW,
    sg0,
    sg1,
    sm0,
    sv0,
    sw0,
    sb0,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    nhw = offsets // HW
    n = nhw // C
    c = nhw % C

    mean = tl.load(mean_ptr + c * sm0, mask=mask, other=0).to(tl.float32)
    var = tl.load(var_ptr + c * sv0, mask=mask, other=0).to(tl.float32)
    gamma = tl.load(weight_ptr + c * sw0, mask=mask, other=0).to(tl.float32)
    beta = tl.load(bias_ptr + c * sb0, mask=mask, other=0).to(tl.float32)
    gate = tl.load(gate_ptr + n * sg0 + c * sg1, mask=mask, other=0).to(tl.float32)

    inv_std = tl.rsqrt(var + 1.0e-5)
    scale = gate * gamma * inv_std
    shift = beta - mean * gamma * inv_std

    x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float32)
    y = x * scale + shift
    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def mul_bn_silu_kernel(
    x_ptr,
    gate_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    C,
    HW,
    sg0,
    sg1,
    sm0,
    sv0,
    sw0,
    sb0,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    nhw = offsets // HW
    n = nhw // C
    c = nhw % C

    mean = tl.load(mean_ptr + c * sm0, mask=mask, other=0).to(tl.float32)
    var = tl.load(var_ptr + c * sv0, mask=mask, other=0).to(tl.float32)
    gamma = tl.load(weight_ptr + c * sw0, mask=mask, other=0).to(tl.float32)
    beta = tl.load(bias_ptr + c * sb0, mask=mask, other=0).to(tl.float32)
    gate = tl.load(gate_ptr + n * sg0 + c * sg1, mask=mask, other=0).to(tl.float32)

    inv_std = tl.rsqrt(var + 1.0e-5)
    scale = gate * gamma * inv_std
    shift = beta - mean * gamma * inv_std

    x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float32)
    y = x * scale + shift
    y = y * tl.sigmoid(y)
    tl.store(out_ptr + offsets, y, mask=mask)



def _route_mul(gate, x):
    out = torch.empty_like(x)
    n_elements = x.numel()
    c = x.shape[1]
    hw = x.shape[2] * x.shape[3]
    block_size = 1024
    grid = (triton.cdiv(n_elements, block_size),)
    mul_broadcast_kernel[grid](
        x,
        gate,
        out,
        n_elements,
        c,
        hw,
        gate.stride(0),
        gate.stride(1),
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return out



def _route_bn_silu(mean, var, bias, weight, x):
    mean = _get_cached_param(mean, x.device)
    var = _get_cached_param(var, x.device)
    bias = _get_cached_param(bias, x.device)
    weight = _get_cached_param(weight, x.device)
    out = torch.empty_like(x)

    n_elements = x.numel()
    c = x.shape[1]
    hw = x.shape[2] * x.shape[3]
    if x.dtype == torch.float32:
        block_size = 256
        num_warps = 4
    else:
        block_size = 512
        num_warps = 8
    grid = (triton.cdiv(n_elements, block_size),)
    bn_silu_kernel[grid](
        x,
        mean,
        var,
        weight,
        bias,
        out,
        n_elements,
        c,
        hw,
        mean.stride(0),
        var.stride(0),
        weight.stride(0),
        bias.stride(0),
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out



def _route_mul_bn(mean, var, bias, weight, gate, x):
    mean = _get_cached_param(mean, x.device)
    var = _get_cached_param(var, x.device)
    bias = _get_cached_param(bias, x.device)
    weight = _get_cached_param(weight, x.device)
    out = torch.empty_like(x)

    n_elements = x.numel()
    c = x.shape[1]
    hw = x.shape[2] * x.shape[3]
    if x.dtype == torch.float32:
        block_size = 256
        num_warps = 4
    else:
        block_size = 512
        num_warps = 8
    grid = (triton.cdiv(n_elements, block_size),)
    mul_bn_kernel[grid](
        x,
        gate,
        mean,
        var,
        weight,
        bias,
        out,
        n_elements,
        c,
        hw,
        gate.stride(0),
        gate.stride(1),
        mean.stride(0),
        var.stride(0),
        weight.stride(0),
        bias.stride(0),
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out



def _route_full(mean, var, bias, weight, gate, x):
    mean = _get_cached_param(mean, x.device)
    var = _get_cached_param(var, x.device)
    bias = _get_cached_param(bias, x.device)
    weight = _get_cached_param(weight, x.device)
    out = torch.empty_like(x)

    n_elements = x.numel()
    c = x.shape[1]
    hw = x.shape[2] * x.shape[3]
    if x.dtype == torch.float32:
        block_size = 256
        num_warps = 4
    else:
        block_size = 512
        num_warps = 8
    grid = (triton.cdiv(n_elements, block_size),)
    mul_bn_silu_kernel[grid](
        x,
        gate,
        mean,
        var,
        weight,
        bias,
        out,
        n_elements,
        c,
        hw,
        gate.stride(0),
        gate.stride(1),
        mean.stride(0),
        var.stride(0),
        weight.stride(0),
        bias.stride(0),
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out


@torch.fx.wrap
def dispatch_replacement(*args):
    route = args[-1]
    if route == "full":
        return _route_full(*args[:-1])
    if route == "bn_silu":
        return _route_bn_silu(*args[:-1])
    if route == "mul_bn":
        return _route_mul_bn(*args[:-1])
    if route == "mul":
        return _route_mul(*args[:-1])
    raise ValueError(f"Unknown route: {route}")