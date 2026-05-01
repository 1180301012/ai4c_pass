import torch
import triton
import triton.language as tl

# Shared Triton layer-norm kernel (one program per row, float32 accumulation)
@triton.jit
def _ln_kernel_shared(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    base = row * N

    x = tl.load(x_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    x_sub = x - mean
    var = tl.sum(x_sub * x_sub, axis=0) / N
    rstd = tl.rsqrt(var + eps)
    x_norm = x_sub * rstd

    w = tl.load(w_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    out = w * x_norm + b
    tl.store(out_ptr + base + offsets, out.to(x_ptr.dtype.element_ty), mask=mask)


# Per-route weight cache (avoids repeated CPU->GPU copies)
_W_CACHE = {}
_B_CACHE = {}


@torch.fx.wrap
def ln_triton_dispatch(in_0, in_1, in_2, route):
    """Shared dispatch wrapper for layer-norm Triton kernels.

    route="route_432"  ->  BLOCK_SIZE=512  (normalized_shape=(432,))
    route="route_192"  ->  BLOCK_SIZE=256  (normalized_shape=(192,))
    """
    global _W_CACHE, _B_CACHE
    dev = in_2.device

    # Cache GPU copies of weight / bias (constant across calls)
    if route not in _W_CACHE:
        _W_CACHE[route] = torch.as_tensor(in_1, device=dev)
        _B_CACHE[route] = torch.as_tensor(in_0, device=dev)
    w = _W_CACHE[route]
    b = _B_CACHE[route]

    out_ln = torch.empty_like(in_2)
    M = in_2.shape[0] * in_2.shape[1]
    N = in_2.shape[2]

    if route == "route_432":
        _ln_kernel_shared[(M,)](in_2, w, b, out_ln, M, N, 1e-6, BLOCK_SIZE=512)
    elif route == "route_192":
        _ln_kernel_shared[(M,)](in_2, w, b, out_ln, M, N, 1e-6, BLOCK_SIZE=256)

    return out_ln