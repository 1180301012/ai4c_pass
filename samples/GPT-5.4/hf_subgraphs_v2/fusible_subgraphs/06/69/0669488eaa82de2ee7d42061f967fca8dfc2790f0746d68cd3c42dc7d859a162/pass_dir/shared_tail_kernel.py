import torch
import triton
import triton.language as tl


@triton.jit
def _tail_kernel(
    x_ptr,
    norm_ptr,
    g_ptr,
    out_ptr,
    M,
    K,
    scale,
    clamp_min,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    base_x = x_ptr + row * K
    base_o = out_ptr + row * K
    n = tl.load(norm_ptr + row).to(tl.float32)
    denom = tl.maximum(n * scale, clamp_min)
    g = tl.load(g_ptr).to(tl.float32)
    mul = g / denom

    mask = offs < K
    x = tl.load(base_x + offs, mask=mask, other=0.0)
    y = x.to(tl.float32) * mul
    tl.store(base_o + offs, y, mask=mask)


@triton.jit
def _tail_kernel_loop(
    x_ptr,
    norm_ptr,
    g_ptr,
    out_ptr,
    M,
    K,
    scale,
    clamp_min,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    base_x = x_ptr + row * K
    base_o = out_ptr + row * K
    n = tl.load(norm_ptr + row).to(tl.float32)
    denom = tl.maximum(n * scale, clamp_min)
    g = tl.load(g_ptr).to(tl.float32)
    mul = g / denom

    k = 0
    while k < K:
        cur = k + offs
        mask = cur < K
        x = tl.load(base_x + cur, mask=mask, other=0.0)
        y = x.to(tl.float32) * mul
        tl.store(base_o + cur, y, mask=mask)
        k += BLOCK_K


@torch.fx.wrap
def fused_tail_dispatch(flattened_x, norm_keepdim, g, route):
    if route == "tail_scale_014433756729740643":
        scale = 0.14433756729740643
    elif route == "tail_scale_007216878364870322":
        scale = 0.07216878364870322
    else:
        raise RuntimeError("unknown route")

    x = flattened_x
    norm = norm_keepdim
    M = x.shape[0] * x.shape[1]
    K = x.shape[2]
    out = torch.empty_like(x)

    # Flatten logical [N, C, K] into [M, K]. Both x and out are contiguous row-major.
    if K <= 256:
        block = 256
        _tail_kernel[(M,)](
            x,
            norm,
            g,
            out,
            M,
            K,
            scale,
            1.0e-5,
            BLOCK_K=block,
        )
    else:
        block = 256
        _tail_kernel_loop[(M,)](
            x,
            norm,
            g,
            out,
            M,
            K,
            scale,
            1.0e-5,
            BLOCK_K=block,
        )
    return out