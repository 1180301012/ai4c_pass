import torch
import triton
import triton.language as tl


# Pattern matching function
# NOTE: This mirrors model.py exactly in op sequence / call style.
def pattern(in_1, in_2):
    tmp_0 = in_2.sigmoid()
    tmp_1 = tmp_0.view(1, -1, 1, 1)
    tmp_2 = tmp_1.expand_as(in_1)
    tmp_3 = in_1 * tmp_2
    return tmp_3


# Argument extraction function
def replacement_args(in_1, in_2):
    return (in_1, in_2)


@triton.jit
def _sigmoid_view_expand_mul_kernel(
    in1_ptr,
    gate_ptr,
    out_ptr,
    C,
    HW,
    W,
    stride1_c,
    stride1_h,
    stride1_w,
    strideg_c,
    out_stride_c,
    out_stride_h,
    out_stride_w,
    BLOCK_HW: tl.constexpr,
):
    pid_c = tl.program_id(0)
    if pid_c >= C:
        return

    hw_offsets = tl.arange(0, BLOCK_HW)
    hw_mask = hw_offsets < HW

    gate = tl.load(gate_ptr + pid_c * strideg_c).to(tl.float32)
    gate = 1.0 / (1.0 + tl.exp(-gate))

    h_offsets = hw_offsets // W
    w_offsets = hw_offsets - h_offsets * W

    in1_base = pid_c * stride1_c
    out_base = pid_c * out_stride_c
    in1_offs = in1_base + h_offsets * stride1_h + w_offsets * stride1_w
    out_offs = out_base + h_offsets * out_stride_h + w_offsets * out_stride_w

    x = tl.load(in1_ptr + in1_offs, mask=hw_mask, other=0.0).to(tl.float32)
    y = x * gate
    tl.store(out_ptr + out_offs, y, mask=hw_mask)


@torch.fx.wrap
def fused_sigmoid_view_expand_mul(in_1, in_2):
    if in_1.ndim != 4 or in_2.ndim != 3:
        raise RuntimeError("Expected in_1 as 4D and in_2 as 3D")
    if in_1.shape[0] != 1 or in_2.shape[0] != 1 or in_2.shape[1] != 1:
        raise RuntimeError("This fused kernel expects batch size 1 and gate shape [1, 1, C]")

    C = in_1.shape[1]
    H = in_1.shape[2]
    W = in_1.shape[3]
    HW = H * W

    if in_2.shape[2] != C:
        raise RuntimeError("Gate channel dimension must match feature channels")
    if HW > 144:
        raise RuntimeError("Spatial size exceeds this fused kernel specialization")

    out = torch.empty_like(in_1)

    if HW <= 49:
        block_hw = 64
        num_warps = 2
    elif HW <= 121:
        block_hw = 128
        num_warps = 4
    else:
        block_hw = 256
        num_warps = 4

    grid = (C,)
    _sigmoid_view_expand_mul_kernel[grid](
        in_1,
        in_2,
        out,
        C,
        HW,
        W,
        in_1.stride(1),
        in_1.stride(2),
        in_1.stride(3),
        in_2.stride(2),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        BLOCK_HW=block_hw,
        num_warps=num_warps,
    )
    return out


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_sigmoid_view_expand_mul