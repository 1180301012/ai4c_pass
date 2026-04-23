import torch
import triton
import triton.language as tl
from graph_net_bench.torch.posion_dispatch_tensor import unwrap_tensor


_AUTOTUNE_CONFIGS = [
    triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
]


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=['H'])
@triton.jit
def _gemma_rmsnorm_suffix_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    H: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_start = row * H

    sumsq = 0.0
    for start in tl.static_range(0, H, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=0).to(tl.float32)
        sumsq += tl.sum(x * x, axis=0)

    inv_rms = tl.rsqrt(sumsq / H + EPS)

    for start in tl.static_range(0, H, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0).to(tl.float32)
        y = x * inv_rms * (1.0 + w)
        tl.store(out_ptr + row_start + cols, y.to(tl.bfloat16), mask=mask)


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=['H'])
@triton.jit
def _gemma_rmsnorm_full_kernel(
    x_ptr,
    w_ptr,
    scale_ptr,
    tmp2_ptr,
    out_ptr,
    H: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_start = row * H
    scale = tl.load(scale_ptr + 0).to(tl.float32)

    sumsq = 0.0
    for start in tl.static_range(0, H, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=0).to(tl.float32)
        prod_bf16 = (x * scale).to(tl.bfloat16)
        prod = prod_bf16.to(tl.float32)
        tl.store(tmp2_ptr + row_start + cols, prod_bf16, mask=mask)
        sumsq += tl.sum(prod * prod, axis=0)

    inv_rms = tl.rsqrt(sumsq / H + EPS)

    for start in tl.static_range(0, H, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=0).to(tl.float32)
        prod = ((x * scale).to(tl.bfloat16)).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0).to(tl.float32)
        y = prod * inv_rms * (1.0 + w)
        tl.store(out_ptr + row_start + cols, y.to(tl.bfloat16), mask=mask)


@torch.fx.wrap
def _gemma_rmsnorm_suffix(x, weight):
    x = unwrap_tensor(x)
    weight = unwrap_tensor(weight)
    hidden_size = weight.numel()
    n_rows = x.numel() // hidden_size
    out = torch.empty_like(x)
    _gemma_rmsnorm_suffix_kernel[(n_rows,)](
        x,
        weight,
        out,
        H=hidden_size,
        EPS=1e-6,
    )
    return out


@torch.fx.wrap
def _gemma_rmsnorm_full(x, weight, scale):
    x = unwrap_tensor(x)
    weight = unwrap_tensor(weight)
    scale = unwrap_tensor(scale)
    hidden_size = weight.numel()
    n_rows = x.numel() // hidden_size
    tmp2 = torch.empty_like(x)
    out = torch.empty_like(x)
    _gemma_rmsnorm_full_kernel[(n_rows,)](
        x,
        weight,
        scale,
        tmp2,
        out,
        H=hidden_size,
        EPS=1e-6,
    )
    return tmp2, out


@torch.fx.wrap
def gemma_rmsnorm_dispatch(*args):
    route = args[-1]
    if route == 'full':
        x, weight, scale, _ = args
        return _gemma_rmsnorm_full(x, weight, scale)
    if route == 'suffix':
        x, weight, _ = args
        return _gemma_rmsnorm_suffix(x, weight)
    raise RuntimeError(f'Unknown route: {route}')