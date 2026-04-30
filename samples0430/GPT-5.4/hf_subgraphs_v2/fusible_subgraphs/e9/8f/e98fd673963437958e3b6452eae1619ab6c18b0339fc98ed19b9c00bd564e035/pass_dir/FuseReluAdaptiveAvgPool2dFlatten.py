import torch
import triton
import triton.language as tl


# Pattern matching function
# NOTE: This mirrors model.py exactly in op sequence / call style.
def pattern(tmp_4):
    tmp_5 = torch.nn.functional.relu(tmp_4, inplace=True)
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(tmp_5, 1)
    tmp_7 = tmp_6.flatten(1, -1)
    return tmp_7


# Argument extraction function
def replacement_args(tmp_4):
    return (tmp_4,)


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
def _relu_pool_flatten_kernel(
    in_ptr,
    out_ptr,
    C,
    HW,
    W,
    stride_c,
    stride_h,
    stride_w,
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
        offs = (
            c_offsets[:, None] * stride_c
            + h_offsets[None, :] * stride_h
            + w_offsets[None, :] * stride_w
        )
        mask = c_mask[:, None] & hw_mask[None, :]

        x = tl.load(in_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        x = tl.maximum(x, 0.0)
        acc += tl.sum(x, axis=1)

    avg = acc / tl.cast(HW, tl.float32)
    tl.store(out_ptr + c_offsets * out_stride_c, avg, mask=c_mask)


@torch.fx.wrap
def fused_relu_adaptive_avg_pool2d_flatten(tmp_4):
    if tmp_4.ndim != 4:
        raise RuntimeError("Expected a 4D input tensor")
    if tmp_4.shape[0] != 1:
        raise RuntimeError("This fused kernel expects batch size 1")

    C = tmp_4.shape[1]
    H = tmp_4.shape[2]
    W = tmp_4.shape[3]
    HW = H * W

    if HW > 144:
        raise RuntimeError("Spatial size exceeds this fused kernel specialization")

    out = torch.empty((1, C), device=tmp_4.device, dtype=tmp_4.dtype)

    grid = lambda META: (triton.cdiv(C, META["BLOCK_C"]),)
    _relu_pool_flatten_kernel[grid](
        tmp_4,
        out,
        C,
        HW,
        W,
        tmp_4.stride(1),
        tmp_4.stride(2),
        tmp_4.stride(3),
        out.stride(1),
    )
    return out


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_relu_adaptive_avg_pool2d_flatten