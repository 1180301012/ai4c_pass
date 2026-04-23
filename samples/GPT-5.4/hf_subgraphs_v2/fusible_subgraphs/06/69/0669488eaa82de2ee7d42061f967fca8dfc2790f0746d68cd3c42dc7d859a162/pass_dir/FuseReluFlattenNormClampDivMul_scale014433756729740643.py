import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_1 = torch.nn.functional.relu(in_1, inplace=True)
    tmp_2 = torch.flatten(tmp_1, 2)
    tmp_3 = torch.functional.norm(tmp_2, 'fro', -1, True, None, None)
    tmp_4 = tmp_3 * 0.14433756729740643
    tmp_5 = tmp_4.clamp(min=1e-05)
    tmp_6 = tmp_2 / tmp_5
    tmp_7 = tmp_6 * in_0
    return (tmp_7,)


def replacement_args(in_0, in_1):
    return (in_0, in_1, "scale_014433756729740643")


@triton.jit
def _relu_l2sum_kernel(
    x_ptr,
    l2_ptr,
    M,
    K,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    base = x_ptr + row * K
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    k = 0
    while k < K:
        cur = k + offs
        mask = cur < K
        x = tl.load(base + cur, mask=mask, other=0.0)
        x = tl.maximum(x, 0)
        xf = x.to(tl.float32)
        acc += xf * xf
        k += BLOCK_K
    s = tl.sum(acc, axis=0)
    tl.store(l2_ptr + row, tl.sqrt(s))


@triton.jit
def _normalize_store_kernel(
    x_ptr,
    l2_ptr,
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
    l2 = tl.load(l2_ptr + row).to(tl.float32)
    denom = tl.maximum(l2 * scale, clamp_min)
    g = tl.load(g_ptr).to(tl.float32)
    mul = g / denom
    k = 0
    while k < K:
        cur = k + offs
        mask = cur < K
        x = tl.load(base_x + cur, mask=mask, other=0.0)
        x = tl.maximum(x, 0)
        y = x.to(tl.float32) * mul
        tl.store(base_o + cur, y, mask=mask)
        k += BLOCK_K


@torch.fx.wrap
def fused_relu_flatten_norm_clamp_div_mul_dispatch(in_0, in_1, route):
    if route == "scale_014433756729740643":
        scale = 0.14433756729740643
    elif route == "scale_007216878364870322":
        scale = 0.07216878364870322
    else:
        raise RuntimeError("unknown route")

    x = in_1
    N, C, H, W = x.shape
    K = H * W
    M = N * C

    out = torch.empty((N, C, K), device=x.device, dtype=x.dtype)
    l2 = torch.empty((M,), device=x.device, dtype=torch.float32)

    BLOCK_K = 256
    _relu_l2sum_kernel[(M,)](
        x,
        l2,
        M,
        K,
        BLOCK_K=BLOCK_K,
    )
    _normalize_store_kernel[(M,)](
        x,
        l2,
        in_0,
        out,
        M,
        K,
        scale,
        1.0e-5,
        BLOCK_K=BLOCK_K,
    )
    return out


def replacement_func():
    return fused_relu_flatten_norm_clamp_div_mul_dispatch