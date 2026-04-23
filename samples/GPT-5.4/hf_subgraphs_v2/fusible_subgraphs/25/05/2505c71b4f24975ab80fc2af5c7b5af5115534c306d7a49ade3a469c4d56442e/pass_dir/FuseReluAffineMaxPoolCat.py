import operator
import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = torch.ops.aten.relu.default(in_2)
    tmp_3 = torch.ops.aten.mul.Tensor(in_1, tmp_2)
    tmp_4 = torch.ops.aten.add.Tensor(tmp_3, in_0)
    tmp_5_with_idx = torch.ops.aten.max_pool2d_with_indices.default(in_3, [2, 2], [1, 1], [0, 0], [1, 1], True)
    tmp_5 = operator.getitem(tmp_5_with_idx, 0)
    tmp_6 = torch.ops.aten.cat.default([tmp_5, tmp_4], 1)
    return (tmp_6,)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


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

def _fused_relu_affine_maxpool_cat_kernel(
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

    # Left branch: max_pool2d(kernel=2, stride=1, padding=0, dilation=1, ceil_mode=True)
    pool_base = ((n * C_LEFT + c) * H_POOL + h) * W_POOL + w
    p00 = tl.load(pool_in_ptr + pool_base, mask=left_valid, other=-float("inf"))
    p01 = tl.load(pool_in_ptr + pool_base + 1, mask=left_valid, other=-float("inf"))
    p10 = tl.load(pool_in_ptr + pool_base + W_POOL, mask=left_valid, other=-float("inf"))
    p11 = tl.load(pool_in_ptr + pool_base + W_POOL + 1, mask=left_valid, other=-float("inf"))
    pool_val = tl.maximum(tl.maximum(p00, p01), tl.maximum(p10, p11))

    # Right branch: relu -> mul(scale) -> add(bias)
    c_right = c - C_LEFT
    x_base = ((n * C_RIGHT + c_right) * H_OUT + h) * W_OUT + w
    x = tl.load(x_ptr + x_base, mask=right_mask, other=0.0)
    x = tl.maximum(x, 0)
    affine_val = x * scale + bias

    out_val = tl.where(left_mask, pool_val, affine_val)
    tl.store(out_ptr + offsets, out_val, mask=mask)


@torch.fx.wrap
def fused_relu_affine_maxpool_cat(in_0, in_1, in_2, in_3):
    N = in_2.shape[0]
    C_RIGHT = in_2.shape[1]
    H_OUT = in_2.shape[2]
    W_OUT = in_2.shape[3]

    C_LEFT = in_3.shape[1]
    H_POOL = in_3.shape[2]
    W_POOL = in_3.shape[3]

    out = torch.empty((N, C_LEFT + C_RIGHT, H_OUT, W_OUT), device=in_2.device, dtype=in_2.dtype)
    n_elements = out.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _fused_relu_affine_maxpool_cat_kernel[grid](
        in_0,
        in_1,
        in_2,
        in_3,
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
    return (out,)


def replacement_func():
    return fused_relu_affine_maxpool_cat