import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_2 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_3 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_4 = torch.cat([tmp_0, tmp_1, tmp_2, tmp_3], 1)
    return tmp_4


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def fused_relu_triple_identical_maxpool_cat_kernel(
    x_ptr,
    out_ptr,
    n_size,
    c_size,
    h_size,
    w_size,
    x_s0,
    x_s1,
    x_s2,
    x_s3,
    o_s0,
    o_s1,
    o_s2,
    o_s3,
    BLOCK_HW: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    hw_total = h_size * w_size
    mask = offs < hw_total

    h = offs // w_size
    w = offs % w_size

    x_base = x_ptr + pid_n * x_s0 + pid_c * x_s1
    o_base0 = out_ptr + pid_n * o_s0 + pid_c * o_s1
    o_base1 = out_ptr + pid_n * o_s0 + (pid_c + c_size) * o_s1
    o_base2 = out_ptr + pid_n * o_s0 + (pid_c + 2 * c_size) * o_s1
    o_base3 = out_ptr + pid_n * o_s0 + (pid_c + 3 * c_size) * o_s1

    center_ptr = x_base + h * x_s2 + w * x_s3
    center = tl.load(center_ptr, mask=mask, other=0.0)
    relu_center = tl.maximum(center, 0)
    out_ptrs0 = o_base0 + h * o_s2 + w * o_s3
    tl.store(out_ptrs0, relu_center, mask=mask)

    pooled = tl.zeros((BLOCK_HW,), dtype=relu_center.dtype)

    for dh in range(-2, 3):
        ih = h + dh
        ih_clamped = tl.maximum(0, tl.minimum(ih, h_size - 1))
        valid_h = (ih >= 0) & (ih < h_size)
        for dw in range(-2, 3):
            iw = w + dw
            iw_clamped = tl.maximum(0, tl.minimum(iw, w_size - 1))
            valid = mask & valid_h & (iw >= 0) & (iw < w_size)
            vals = tl.load(x_base + ih_clamped * x_s2 + iw_clamped * x_s3, mask=valid, other=0.0)
            vals = tl.maximum(vals, 0)
            pooled = tl.maximum(pooled, vals)

    out_ptrs1 = o_base1 + h * o_s2 + w * o_s3
    out_ptrs2 = o_base2 + h * o_s2 + w * o_s3
    out_ptrs3 = o_base3 + h * o_s2 + w * o_s3
    tl.store(out_ptrs1, pooled, mask=mask)
    tl.store(out_ptrs2, pooled, mask=mask)
    tl.store(out_ptrs3, pooled, mask=mask)


@torch.fx.wrap
def fused_relu_triple_identical_maxpool_cat(in_0):
    if not in_0.is_cuda:
        raise RuntimeError("fused_relu_triple_identical_maxpool_cat expects a CUDA tensor")
    if in_0.ndim != 4:
        raise RuntimeError("fused_relu_triple_identical_maxpool_cat expects NCHW input")

    n_size = in_0.shape[0]
    c_size = in_0.shape[1]
    h_size = in_0.shape[2]
    w_size = in_0.shape[3]

    out = torch.empty((n_size, c_size * 4, h_size, w_size), device=in_0.device, dtype=in_0.dtype)

    block_hw = 64
    grid = (triton.cdiv(h_size * w_size, block_hw), c_size, n_size)
    fused_relu_triple_identical_maxpool_cat_kernel[grid](
        in_0,
        out,
        n_size,
        c_size,
        h_size,
        w_size,
        in_0.stride(0),
        in_0.stride(1),
        in_0.stride(2),
        in_0.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        BLOCK_HW=block_hw,
        num_warps=4,
        num_stages=2,
    )
    return out


def replacement_func():
    return fused_relu_triple_identical_maxpool_cat