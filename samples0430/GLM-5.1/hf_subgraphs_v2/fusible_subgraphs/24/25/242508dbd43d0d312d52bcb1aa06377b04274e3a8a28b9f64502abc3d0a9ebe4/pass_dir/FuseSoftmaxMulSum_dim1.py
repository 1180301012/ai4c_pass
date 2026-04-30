import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = torch.softmax(in_1, dim=1)
    tmp_1 = in_0 * tmp_0
    tmp_2 = torch.sum(tmp_1, dim=1)
    return (tmp_2,)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_softmax_mul_sum_kernel(
    in_0_ptr, in_1_ptr, out_ptr,
    C, HW, W,
    in_0_stride1, in_0_stride2, in_0_stride3, in_0_stride4,
    in_1_stride1, in_1_stride2,
    out_stride1, out_stride2, out_stride3,
    SPATIAL_BLOCK: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_s = tl.program_id(1)

    c = pid_c
    s_offsets = pid_s * SPATIAL_BLOCK + tl.arange(0, SPATIAL_BLOCK)
    s_mask = s_offsets < HW

    h = s_offsets // W
    w = s_offsets % W

    # Load softmax input (2 scalar values for this channel)
    v0 = tl.load(in_1_ptr + c * in_1_stride2)
    v1 = tl.load(in_1_ptr + in_1_stride1 + c * in_1_stride2)

    # Softmax in float32 for numerical stability
    v0_f = v0.to(tl.float32)
    v1_f = v1.to(tl.float32)
    max_val = tl.maximum(v0_f, v1_f)
    exp_v0 = tl.exp(v0_f - max_val)
    exp_v1 = tl.exp(v1_f - max_val)
    sum_exp = exp_v0 + exp_v1
    w0 = exp_v0 / sum_exp
    w1 = exp_v1 / sum_exp

    # Load in_0 values for both channels (k=0 and k=1 in dim=1)
    base_offset = c * in_0_stride2 + h * in_0_stride3 + w * in_0_stride4
    a0 = tl.load(in_0_ptr + base_offset, mask=s_mask, other=0.0)
    a1 = tl.load(in_0_ptr + in_0_stride1 + base_offset, mask=s_mask, other=0.0)

    # Compute weighted sum (w0, w1 are scalars that broadcast to vector)
    out_val = a0.to(tl.float32) * w0 + a1.to(tl.float32) * w1

    # Store result
    out_offset = c * out_stride1 + h * out_stride2 + w * out_stride3
    tl.store(out_ptr + out_offset, out_val, mask=s_mask)


@torch.fx.wrap
def fused_softmax_mul_sum(in_0, in_1):
    C = in_0.shape[2]
    H = in_0.shape[3]
    W_val = in_0.shape[4]
    HW = H * W_val

    out_shape = [in_0.shape[0], C, H, W_val]
    out = torch.empty(out_shape, dtype=in_0.dtype, device=in_0.device)

    SPATIAL_BLOCK = 256
    grid = (C, (HW + SPATIAL_BLOCK - 1) // SPATIAL_BLOCK)

    fused_softmax_mul_sum_kernel[grid](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        C=C, HW=HW, W=W_val,
        in_0_stride1=in_0.stride(1),
        in_0_stride2=in_0.stride(2),
        in_0_stride3=in_0.stride(3),
        in_0_stride4=in_0.stride(4),
        in_1_stride1=in_1.stride(1),
        in_1_stride2=in_1.stride(2),
        out_stride1=out.stride(1),
        out_stride2=out.stride(2),
        out_stride3=out.stride(3),
        SPATIAL_BLOCK=SPATIAL_BLOCK,
    )

    return (out,)


def replacement_func():
    return fused_softmax_mul_sum