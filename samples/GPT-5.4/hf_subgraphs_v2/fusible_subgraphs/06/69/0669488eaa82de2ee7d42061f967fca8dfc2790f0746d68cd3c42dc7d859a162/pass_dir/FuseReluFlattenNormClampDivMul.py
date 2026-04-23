import torch
import triton
import triton.language as tl


# Match: relu(inplace=True) -> flatten(2) -> norm(dim=-1, keepdim=True)
#      -> mul(scale) -> clamp(min=1e-05) -> div -> mul(g)
# The scale is kept symbolic to cover both graph variants.
def pattern(in_0, in_1, scale):
    tmp_1 = torch.nn.functional.relu(in_1, inplace=True)
    tmp_2 = torch.flatten(tmp_1, 2)
    tmp_3 = torch.functional.norm(tmp_2, dim=-1, keepdim=True)
    tmp_4 = tmp_3 * scale
    tmp_5 = tmp_4.clamp(min=1e-05)
    tmp_6 = tmp_2 / tmp_5
    tmp_7 = tmp_6 * in_0
    return (tmp_7,)


def replacement_args(in_0, in_1, scale):
    return (in_0, in_1, scale)


@triton.jit
def _relu_l2sum_kernel(
    x_ptr,
    l2_ptr,
    M,
    K,
    stride_m,
    stride_k,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offs_k = tl.arange(0, BLOCK_K)
    base = x_ptr + row * stride_m
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    k = 0
    while k < K:
        cur = k + offs_k
        mask = cur < K
        x = tl.load(base + cur * stride_k, mask=mask, other=0.0)
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
    stride_xm,
    stride_xk,
    stride_om,
    stride_ok,
    scale,
    clamp_min,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offs_k = tl.arange(0, BLOCK_K)
    base_x = x_ptr + row * stride_xm
    base_o = out_ptr + row * stride_om
    l2 = tl.load(l2_ptr + row).to(tl.float32)
    denom = l2 * scale
    denom = tl.maximum(denom, clamp_min)
    g = tl.load(g_ptr).to(tl.float32)
    inv = g / denom
    k = 0
    while k < K:
        cur = k + offs_k
        mask = cur < K
        x = tl.load(base_x + cur * stride_xk, mask=mask, other=0.0)
        x = tl.maximum(x, 0)
        y = x.to(tl.float32) * inv
        tl.store(base_o + cur * stride_ok, y, mask=mask)
        k += BLOCK_K


@torch.fx.wrap
def fused_relu_flatten_norm_clamp_div_mul(in_0, in_1, scale):
    # Input layout after flatten(2): [M, K], where M = prod(shape[:-1] after flatten) and K = last dim.
    # Since flatten(2) on contiguous NCHW only merges the last two dimensions, the result is viewable as [N*C, H*W].
    # We operate directly on the original storage using explicit strides to avoid materializing the flattened relu tensor.
    x = in_1
    g = in_0
    M = x.shape[0] * x.shape[1]
    K = x.shape[2] * x.shape[3]

    out = torch.empty((x.shape[0], x.shape[1], K), device=x.device, dtype=x.dtype)
    l2 = torch.empty((M,), device=x.device, dtype=torch.float32)

    stride_xm = x.stride(0) * x.shape[1]
    # For contiguous [N,C,H,W], row-major flatten over the last 2 dims is contiguous.
    # The row base for logical row r = n*C + c is x_ptr + n*stride0 + c*stride1.
    # The kernel receives stride_xm = stride1 in elements and stride_xk = 1, with x_ptr adjusted by row*stride_xm only.
    # That works because contiguous tensors satisfy stride1 == H*W and rows are packed accordingly.
    # More generally for contiguous NCHW this is valid.
    stride_xm = x.shape[2] * x.shape[3]
    stride_xk = 1
    stride_om = K
    stride_ok = 1

    # Use a moderate block size because K is small (48 or 192 in all provided graphs).
    BLOCK_K = 256

    _relu_l2sum_kernel[(M,)](
        x,
        l2,
        M,
        K,
        stride_xm,
        stride_xk,
        BLOCK_K=BLOCK_K,
    )

    _normalize_store_kernel[(M,)](
        x,
        l2,
        g,
        out,
        M,
        K,
        stride_xm,
        stride_xk,
        stride_om,
        stride_ok,
        float(scale),
        1.0e-5,
        BLOCK_K=BLOCK_K,
    )
    return (out,)


def replacement_func():
    return fused_relu_flatten_norm_clamp_div_mul