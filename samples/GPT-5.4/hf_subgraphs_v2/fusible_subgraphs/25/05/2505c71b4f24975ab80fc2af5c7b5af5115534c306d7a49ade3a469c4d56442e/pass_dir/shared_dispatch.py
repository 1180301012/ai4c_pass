import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HW": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_HW": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_HW": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_HW": 2048}, num_warps=8, num_stages=2),
    ],
    key=["HW", "TOTAL_PLANES"],
)
@triton.jit
def _cat_dim1_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    N,
    C_A,
    C_B,
    HW,
    TOTAL_PLANES,
    BLOCK_HW: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_plane = tl.program_id(1)

    offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    hw_mask = offs < HW
    plane_mask = pid_plane < TOTAL_PLANES
    mask = hw_mask & plane_mask

    c_out = pid_plane % (C_A + C_B)
    n = pid_plane // (C_A + C_B)
    from_a = c_out < C_A

    a_plane = n * C_A + c_out
    b_plane = n * C_B + (c_out - C_A)
    src_idx_a = a_plane * HW + offs
    src_idx_b = b_plane * HW + offs
    out_idx = pid_plane * HW + offs

    a_val = tl.load(a_ptr + src_idx_a, mask=mask & from_a, other=0.0)
    b_val = tl.load(b_ptr + src_idx_b, mask=mask & (~from_a), other=0.0)
    out_val = tl.where(from_a, a_val, b_val)
    tl.store(out_ptr + out_idx, out_val, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
    ],
    key=["n_elements"],
)
@triton.jit
def _add_broadcast_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    a_numel,
    b_numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a_idx = tl.where(a_numel == 1, 0, offsets)
    b_idx = tl.where(b_numel == 1, 0, offsets)

    a = tl.load(a_ptr + a_idx, mask=mask, other=0.0)
    b = tl.load(b_ptr + b_idx, mask=mask, other=0.0)
    out = a + b
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
    ],
    key=["n_elements"],
)
@triton.jit
def _relu_affine_kernel(
    bias_ptr,
    scale_ptr,
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    bias = tl.load(bias_ptr)
    scale = tl.load(scale_ptr)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x = tl.maximum(x, 0)
    out = x * scale + bias
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
    ],
    key=["n_elements"],
)
@triton.jit
def _maxpool2x2s1_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    N,
    C,
    H_OUT,
    W_OUT,
    H_IN,
    W_IN,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    hw = H_OUT * W_OUT
    chw = C * hw

    n = offsets // chw
    rem0 = offsets % chw
    c = rem0 // hw
    rem1 = rem0 % hw
    h = rem1 // W_OUT
    w = rem1 % W_OUT

    base = ((n * C + c) * H_IN + h) * W_IN + w
    p00 = tl.load(x_ptr + base, mask=mask, other=-float("inf"))
    p01 = tl.load(x_ptr + base + 1, mask=mask, other=-float("inf"))
    p10 = tl.load(x_ptr + base + W_IN, mask=mask, other=-float("inf"))
    p11 = tl.load(x_ptr + base + W_IN + 1, mask=mask, other=-float("inf"))
    out = tl.maximum(tl.maximum(p00, p01), tl.maximum(p10, p11))
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
    ],
    key=["n_elements"],
)
@triton.jit
def _whole_fused_kernel(
    bias_ptr,
    scale_ptr,
    x_ptr,
    pool_in_ptr,
    out_ptr,
    n_elements,
    N,
    C_LEFT,
    C_RIGHT,
    H_OUT,
    W_OUT,
    H_POOL,
    W_POOL,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    hw = H_OUT * W_OUT
    chw = (C_LEFT + C_RIGHT) * hw

    n = offsets // chw
    rem0 = offsets % chw
    c = rem0 // hw
    rem1 = rem0 % hw
    h = rem1 // W_OUT
    w = rem1 % W_OUT

    left_mask = c < C_LEFT
    right_mask = mask & (~left_mask)
    left_valid = mask & left_mask

    bias = tl.load(bias_ptr)
    scale = tl.load(scale_ptr)

    pool_base = ((n * C_LEFT + c) * H_POOL + h) * W_POOL + w
    p00 = tl.load(pool_in_ptr + pool_base, mask=left_valid, other=-float("inf"))
    p01 = tl.load(pool_in_ptr + pool_base + 1, mask=left_valid, other=-float("inf"))
    p10 = tl.load(pool_in_ptr + pool_base + W_POOL, mask=left_valid, other=-float("inf"))
    p11 = tl.load(pool_in_ptr + pool_base + W_POOL + 1, mask=left_valid, other=-float("inf"))
    pool_val = tl.maximum(tl.maximum(p00, p01), tl.maximum(p10, p11))

    c_right = c - C_LEFT
    x_base = ((n * C_RIGHT + c_right) * H_OUT + h) * W_OUT + w
    x = tl.load(x_ptr + x_base, mask=right_mask, other=0.0)
    x = tl.maximum(x, 0)
    affine_val = x * scale + bias

    out_val = tl.where(left_mask, pool_val, affine_val)
    tl.store(out_ptr + offsets, out_val, mask=mask)



@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HW": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_HW": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_HW": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_HW": 2048}, num_warps=8, num_stages=2),
    ],
    key=["HW", "TOTAL_PLANES"],
)
@triton.jit
def _add_cat_right_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    out_ptr,
    B_NUMEL,
    N,
    C_LEFT,
    C_RIGHT,
    HW,
    TOTAL_PLANES,
    BLOCK_HW: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_plane = tl.program_id(1)

    offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    hw_mask = offs < HW
    plane_mask = pid_plane < TOTAL_PLANES
    mask = hw_mask & plane_mask

    c_idx = pid_plane % (C_LEFT + C_RIGHT)
    left_side = c_idx < C_LEFT

    out_base = pid_plane * HW
    out_idx = out_base + offs

    left_base = pid_plane * HW
    left_val = tl.load(c_ptr + left_base + offs, mask=mask & left_side, other=0.0)

    right_plane = pid_plane - C_LEFT - (pid_plane // (C_LEFT + C_RIGHT)) * C_LEFT
    a_base = right_plane * HW
    a_val = tl.load(a_ptr + a_base + offs, mask=mask & (~left_side), other=0.0)

    if B_NUMEL == 1:
        b_val = tl.load(b_ptr)
    else:
        b_val = tl.load(b_ptr + a_base + offs, mask=mask & (~left_side), other=0.0)

    right_val = a_val + b_val
    out_val = tl.where(left_side, left_val, right_val)
    tl.store(out_ptr + out_idx, out_val, mask=mask)


@torch.fx.wrap
def replacement_impl(*args):
    route = args[-1]
    if route == "cat_dim1":
        return _cat_dim1(*args[:-1])
    if route == "add_broadcast":
        return _add_broadcast(*args[:-1])
    if route == "add_cat_right":
        return _add_cat_right(*args[:-1])
    if route == "relu_affine":
        return _relu_affine(*args[:-1])
    if route == "maxpool2x2s1":
        return _maxpool2x2s1(*args[:-1])
    if route == "whole_fused":
        return _whole_fused(*args[:-1])
    raise RuntimeError(f"Unknown route: {route}")


def _launch_grid(n_elements):
    return lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)


def _cat_dim1(a, b):
    N = a.shape[0]
    C_A = a.shape[1]
    C_B = b.shape[1]
    H = a.shape[2]
    W = a.shape[3]
    out = torch.empty((N, C_A + C_B, H, W), device=a.device, dtype=a.dtype)
    HW = H * W
    TOTAL_PLANES = N * (C_A + C_B)
    grid = lambda meta: (triton.cdiv(HW, meta["BLOCK_HW"]), TOTAL_PLANES)
    _cat_dim1_kernel[grid](
        a,
        b,
        out,
        N,
        C_A,
        C_B,
        HW,
        TOTAL_PLANES,
    )
    return out


def _add_broadcast(a, b):
    ref = a if a.numel() >= b.numel() else b
    out = torch.empty_like(ref)
    n_elements = out.numel()
    _add_broadcast_kernel[_launch_grid(n_elements)](
        a,
        b,
        out,
        n_elements,
        a.numel(),
        b.numel(),
    )
    return out


def _relu_affine(bias, scale, x):
    out = torch.empty_like(x)
    n_elements = out.numel()
    _relu_affine_kernel[_launch_grid(n_elements)](
        bias,
        scale,
        x,
        out,
        n_elements,
    )
    return out


def _add_cat_right(a, b, c):
    N = c.shape[0]
    C_LEFT = c.shape[1]
    C_RIGHT = a.shape[1]
    H = a.shape[2]
    W = a.shape[3]
    out = torch.empty((N, C_LEFT + C_RIGHT, H, W), device=a.device, dtype=a.dtype)
    HW = H * W
    TOTAL_PLANES = N * (C_LEFT + C_RIGHT)
    grid = lambda meta: (triton.cdiv(HW, meta["BLOCK_HW"]), TOTAL_PLANES)
    _add_cat_right_kernel[grid](
        a,
        b,
        c,
        out,
        b.numel(),
        N,
        C_LEFT,
        C_RIGHT,
        HW,
        TOTAL_PLANES,
    )
    return out


def _maxpool2x2s1(x):
    N = x.shape[0]
    C = x.shape[1]
    H_IN = x.shape[2]
    W_IN = x.shape[3]
    H_OUT = H_IN - 1
    W_OUT = W_IN - 1
    out = torch.empty((N, C, H_OUT, W_OUT), device=x.device, dtype=x.dtype)
    n_elements = out.numel()
    _maxpool2x2s1_kernel[_launch_grid(n_elements)](
        x,
        out,
        n_elements,
        N,
        C,
        H_OUT,
        W_OUT,
        H_IN,
        W_IN,
    )
    return out


def _whole_fused(bias, scale, x, pool_in):
    N = x.shape[0]
    C_RIGHT = x.shape[1]
    H_OUT = x.shape[2]
    W_OUT = x.shape[3]
    C_LEFT = pool_in.shape[1]
    H_POOL = pool_in.shape[2]
    W_POOL = pool_in.shape[3]
    out = torch.empty((N, C_LEFT + C_RIGHT, H_OUT, W_OUT), device=x.device, dtype=x.dtype)
    n_elements = out.numel()
    _whole_fused_kernel[_launch_grid(n_elements)](
        bias,
        scale,
        x,
        pool_in,
        out,
        n_elements,
        N,
        C_LEFT,
        C_RIGHT,
        H_OUT,
        W_OUT,
        H_POOL,
        W_POOL,
    )
    return out