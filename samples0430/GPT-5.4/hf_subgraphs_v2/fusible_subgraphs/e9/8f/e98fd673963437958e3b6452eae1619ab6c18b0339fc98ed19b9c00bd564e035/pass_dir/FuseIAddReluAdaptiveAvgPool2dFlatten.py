import operator
import torch
import triton
import triton.language as tl


# Pattern matching function
# NOTE: This mirrors the traced graph, including in-place add.
def pattern(tmp_3, in_0):
    if isinstance(tmp_3, torch.fx.Proxy):
        tmp_4 = tmp_3.tracer.create_proxy("call_function", operator.iadd, (tmp_3, in_0), {})
    else:
        tmp_4 = operator.iadd(tmp_3, in_0)
    tmp_5 = torch.nn.functional.relu(tmp_4, inplace=True)
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(tmp_5, 1)
    tmp_7 = tmp_6.flatten(1, -1)
    return tmp_7


# Argument extraction function
def replacement_args(tmp_3, in_0):
    return (tmp_3, in_0)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": 32, "BLOCK_HW": 8}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_C": 32, "BLOCK_HW": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_C": 64, "BLOCK_HW": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_C": 64, "BLOCK_HW": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_C": 128, "BLOCK_HW": 8}, num_warps=8, num_stages=2),
    ],
    key=["C", "HW", "W"],
)
@triton.jit
def _iadd_relu_pool_flatten_kernel(
    x_ptr,
    r_ptr,
    out_ptr,
    C,
    HW,
    W,
    stridex_c,
    stridex_h,
    stridex_w,
    strider_c,
    strider_h,
    strider_w,
    out_stride_c,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    MAX_HW: tl.constexpr = 144,
):
    pid = tl.program_id(0)
    c_offsets = pid * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offsets < C

    acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    hw_offsets_base = tl.arange(0, BLOCK_HW)

    for hw_start in tl.static_range(0, MAX_HW, BLOCK_HW):
        hw_offsets = hw_start + hw_offsets_base
        hw_mask = hw_offsets < HW

        h_offsets = hw_offsets // W
        w_offsets = hw_offsets - h_offsets * W

        offs_x = (
            c_offsets[:, None] * stridex_c
            + h_offsets[None, :] * stridex_h
            + w_offsets[None, :] * stridex_w
        )
        offs_r = (
            c_offsets[:, None] * strider_c
            + h_offsets[None, :] * strider_h
            + w_offsets[None, :] * strider_w
        )
        mask = c_mask[:, None] & hw_mask[None, :]

        x = tl.load(x_ptr + offs_x, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(r_ptr + offs_r, mask=mask, other=0.0).to(tl.float32)
        y = x + r
        y = tl.maximum(y, 0.0)
        acc += tl.sum(y, axis=1)

    avg = acc / tl.cast(HW, tl.float32)
    tl.store(out_ptr + c_offsets * out_stride_c, avg, mask=c_mask)


@torch.fx.wrap
def fused_iadd_relu_adaptive_avg_pool2d_flatten(tmp_3, in_0):
    if tmp_3.ndim != 4 or in_0.ndim != 4:
        raise RuntimeError("Expected two 4D tensors")
    if tmp_3.shape[0] != 1 or in_0.shape[0] != 1:
        raise RuntimeError("This fused kernel expects batch size 1")
    if tmp_3.shape[1] != in_0.shape[1] or tmp_3.shape[2] != in_0.shape[2] or tmp_3.shape[3] != in_0.shape[3]:
        raise RuntimeError("Input tensor shapes must match")

    C = tmp_3.shape[1]
    H = tmp_3.shape[2]
    W = tmp_3.shape[3]
    HW = H * W

    if HW > 144:
        raise RuntimeError("Spatial size exceeds this fused kernel specialization")

    out = torch.empty((1, C), device=tmp_3.device, dtype=tmp_3.dtype)

    grid = lambda META: (triton.cdiv(C, META["BLOCK_C"]),)
    _iadd_relu_pool_flatten_kernel[grid](
        tmp_3,
        in_0,
        out,
        C,
        HW,
        W,
        tmp_3.stride(1),
        tmp_3.stride(2),
        tmp_3.stride(3),
        in_0.stride(1),
        in_0.stride(2),
        in_0.stride(3),
        out.stride(1),
    )
    return out


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_iadd_relu_adaptive_avg_pool2d_flatten