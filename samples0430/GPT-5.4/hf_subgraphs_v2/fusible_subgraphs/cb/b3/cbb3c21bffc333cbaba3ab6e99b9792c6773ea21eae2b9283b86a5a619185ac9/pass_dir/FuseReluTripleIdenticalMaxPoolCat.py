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


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HW": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_HW": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_HW": 256}, num_warps=8, num_stages=1),
    ],
    key=["N"],
)
@triton.jit
def fused_relu_pool_cat_kernel(
    x_ptr,
    out_ptr,
    N,
    BLOCK_HW: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_nc = tl.program_id(1)

    C = 256
    H = 20
    W = 20
    HW = 400
    CHW = 102400
    OCHW = 409600

    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = offs_hw < HW

    n = pid_nc // C
    c = pid_nc % C

    h = offs_hw // W
    w = offs_hw % W

    in_base = x_ptr + n * CHW + c * HW
    center_ptrs = in_base + offs_hw

    center = tl.load(center_ptrs, mask=mask, other=0.0)
    relu_center = tl.maximum(center, 0)
    max_val = relu_center

    for dh in range(-2, 3):
        for dw in range(-2, 3):
            if dh == 0 and dw == 0:
                continue
            nh = h + dh
            nw = w + dw
            nmask = mask & (nh >= 0) & (nh < H) & (nw >= 0) & (nw < W)
            ptrs = in_base + nh * W + nw
            vals = tl.load(ptrs, mask=nmask, other=0.0)
            vals = tl.maximum(vals, 0)
            max_val = tl.maximum(max_val, vals)

    out_base = out_ptr + n * OCHW + c * HW + offs_hw
    tl.store(out_base, relu_center, mask=mask)
    tl.store(out_base + CHW, max_val, mask=mask)
    tl.store(out_base + 2 * CHW, max_val, mask=mask)
    tl.store(out_base + 3 * CHW, max_val, mask=mask)


@torch.fx.wrap
def fused_relu_triple_pool_cat(x):
    n = x.shape[0]
    out = torch.empty((n, 1024, 20, 20), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(400, meta["BLOCK_HW"]), n * 256)
    fused_relu_pool_cat_kernel[grid](x, out, n)
    return out


def replacement_func():
    return fused_relu_triple_pool_cat