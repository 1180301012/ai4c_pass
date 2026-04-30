import torch
import triton
import triton.language as tl


_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
]


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["n_elements"])
@triton.jit
def fused_silu_add_inplace_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)
    silu_x = x_f32 * tl.sigmoid(x_f32)
    out = silu_x + y_f32

    tl.store(x_ptr + offsets, silu_x, mask=mask)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["n_elements"])
@triton.jit
def silu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    out = x_f32 * tl.sigmoid(x_f32)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["n_elements"])
@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x.to(tl.float32) + y.to(tl.float32)
    tl.store(out_ptr + offsets, out, mask=mask)


def _grid_1d(n_elements):
    return lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)


def _launch_fused_silu_add_inplace(in_0, in_1):
    n_elements = in_0.numel()
    out = torch.empty_like(in_0)
    fused_silu_add_inplace_kernel[_grid_1d(n_elements)](in_1, in_0, out, n_elements)
    return out


def _launch_silu(x):
    n_elements = x.numel()
    out = torch.empty_like(x)
    silu_kernel[_grid_1d(n_elements)](x, out, n_elements)
    return out


def _launch_add(x, y):
    n_elements = x.numel()
    out = torch.empty_like(x)
    add_kernel[_grid_1d(n_elements)](x, y, out, n_elements)
    return out


@torch.fx.wrap
def fused_silu_add_dispatch(arg0, arg1, route):
    if route == "out_only":
        return _launch_fused_silu_add_inplace(arg0, arg1)
    if route == "return_input_out":
        out = _launch_fused_silu_add_inplace(arg0, arg1)
        return arg1, out
    if route == "silu_only":
        return _launch_silu(arg0)
    if route == "add_only":
        return _launch_add(arg0, arg1)
    raise RuntimeError(f"Unsupported route: {route}")


def shared_replacement_func():
    return fused_silu_add_dispatch