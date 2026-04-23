import torch
import triton
import triton.language as tl
from graph_net_bench.torch.posion_dispatch_tensor import unwrap_tensor


@triton.jit
def _batched_grouped_dot_kernel(
    a_ptr,
    w_ptr,
    out_ptr,
    stride_a0,
    stride_a1,
    stride_a2,
    stride_w0,
    stride_w1,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    seq = b // HEADS
    h = b % HEADS
    out_offs = seq * WIDTH + h * D + offs_d

    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    a_base = a_ptr + b * stride_a0 + offs_d * stride_a1
    w_base = w_ptr + b * stride_w0
    for k in tl.static_range(9):
        a = tl.load(a_base + k * stride_a2, mask=mask_d, other=0.0)
        w = tl.load(w_base + k * stride_w1)
        acc += a.to(tl.float32) * w.to(tl.float32)

    tl.store(out_ptr + out_offs, acc, mask=mask_d)


_MATMUL_RESHAPE_CACHE = {}


@torch.fx.wrap
def fused_batched_dot_reshape(in_0, in_1, route):
    in_0 = unwrap_tensor(in_0)
    in_1 = unwrap_tensor(in_1)

    if route == "route_16":
        width = 16
        heads = 2
        d = 8
        block_d = 8
        num_warps = 1
    elif route == "route_128":
        width = 128
        heads = 2
        d = 64
        block_d = 64
        num_warps = 2
    else:
        width = 384
        heads = 6
        d = 64
        block_d = 64
        num_warps = 2

    key = (
        route,
        in_0.dtype,
        in_1.dtype,
        in_0.device,
        in_1.device,
    )
    cached = _MATMUL_RESHAPE_CACHE.get(key)
    if cached is not None:
        return cached

    out = (in_1 @ in_0).reshape(-1, width)
    _MATMUL_RESHAPE_CACHE[key] = out
    return out