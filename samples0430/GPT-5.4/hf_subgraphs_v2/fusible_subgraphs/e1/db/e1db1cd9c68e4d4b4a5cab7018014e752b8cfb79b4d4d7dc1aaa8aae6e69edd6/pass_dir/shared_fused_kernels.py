import torch
import triton
import triton.language as tl


@triton.jit
def gelu_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    y_f32 = x_f32 * (0.5 * (1.0 + tl.math.erf(x_f32 * 0.7071067811865475)))
    tl.store(y_ptr + offs, y_f32, mask=mask)


@triton.jit
def spatial_mean_kernel(
    x_ptr,
    out_ptr,
    hw,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * hw
    acc = tl.zeros((), dtype=tl.float32)
    for start in range(0, hw, BLOCK_HW):
        offs = start + tl.arange(0, BLOCK_HW)
        mask = offs < hw
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        acc += tl.sum(tl.where(mask, x.to(tl.float32), 0.0), axis=0)
    tl.store(out_ptr + pid, acc / hw)


@triton.jit
def fused_gelu_mean_kernel(
    x_ptr,
    y_ptr,
    mean_ptr,
    hw,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * hw
    acc = tl.zeros((), dtype=tl.float32)
    for start in range(0, hw, BLOCK_HW):
        offs = start + tl.arange(0, BLOCK_HW)
        mask = offs < hw
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)
        y_f32 = x_f32 * (0.5 * (1.0 + tl.math.erf(x_f32 * 0.7071067811865475)))
        tl.store(y_ptr + base + offs, y_f32, mask=mask)
        acc += tl.sum(tl.where(mask, y_f32, 0.0), axis=0)
    tl.store(mean_ptr + pid, acc / hw)


@torch.fx.wrap
def route_dispatch(x, route):
    if route == 'gelu':
        n_elements = x.numel()
        out = torch.empty_like(x)
        grid = (triton.cdiv(n_elements, 1024),)
        gelu_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024, num_warps=4)
        return out
    if route == 'mean':
        n = x.shape[0]
        c = x.shape[1]
        hw = x.shape[2] * x.shape[3]
        out = torch.empty((n, c, 1, 1), device=x.device, dtype=x.dtype)
        grid = (n * c,)
        spatial_mean_kernel[grid](x, out, hw, BLOCK_HW=256, num_warps=4)
        return out
    if route == 'fused':
        n = x.shape[0]
        c = x.shape[1]
        hw = x.shape[2] * x.shape[3]
        y = torch.empty_like(x)
        mean = torch.empty((n, c, 1, 1), device=x.device, dtype=x.dtype)
        grid = (n * c,)
        fused_gelu_mean_kernel[grid](x, y, mean, hw, BLOCK_HW=256, num_warps=4)
        return (y, mean)
    return x


def shared_replacement_func():
    return route_dispatch