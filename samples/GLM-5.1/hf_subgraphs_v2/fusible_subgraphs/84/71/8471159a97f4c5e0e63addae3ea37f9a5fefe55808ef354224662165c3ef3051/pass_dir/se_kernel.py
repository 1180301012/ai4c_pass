import torch
import triton
import triton.language as tl

@triton.jit
def fused_se_kernel(
    in_2_ptr, in_1_ptr, in_0_ptr, in_3_ptr, out_ptr,
    B, C_in, C_out, H, W,
    in_2_stride0, in_2_stride1,
    in_1_stride0, in_1_stride1,
    in_0_stride0,
    in_3_stride0, in_3_stride1, in_3_stride2, in_3_stride3,
    out_stride0, out_stride1, out_stride2, out_stride3,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    b_idx = pid // C_out
    c_idx = pid % C_out

    # Load bias for this channel (cast to fp32 for computation)
    bias = tl.load(in_0_ptr + c_idx * in_0_stride0).to(tl.float32)

    # Compute linear: dot product of in_2[b, :] with in_1[c, :]
    # Cast inputs to fp32 for accumulation
    acc = bias
    for k in range(C_in):
        x_val = tl.load(in_2_ptr + b_idx * in_2_stride0 + k * in_2_stride1).to(tl.float32)
        w_val = tl.load(in_1_ptr + c_idx * in_1_stride0 + k * in_1_stride1).to(tl.float32)
        acc = acc + x_val * w_val

    # Sigmoid (computed in fp32)
    sigmoid_val = tl.sigmoid(acc)

    # Broadcast multiply over HW
    # Cast sigmoid back to output dtype for the multiply
    n_hw = H * W
    for hw_start in range(0, n_hw, BLOCK_HW):
        hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
        hw_mask = hw_offsets < n_hw
        h_idx = hw_offsets // W
        w_idx = hw_offsets % W

        feat_val = tl.load(
            in_3_ptr + b_idx * in_3_stride0 + c_idx * in_3_stride1 + h_idx * in_3_stride2 + w_idx * in_3_stride3,
            mask=hw_mask, other=0.0
        ).to(tl.float32)
        out_val = (feat_val * sigmoid_val).to(out_ptr.type.element_ty)
        tl.store(
            out_ptr + b_idx * out_stride0 + c_idx * out_stride1 + h_idx * out_stride2 + w_idx * out_stride3,
            out_val, mask=hw_mask
        )


@torch.fx.wrap
def fused_se_dispatch(in_0, in_1, in_2, in_3, route):
    B = in_3.shape[0]
    C_out = in_3.shape[1]
    H = in_3.shape[2]
    W = in_3.shape[3]
    C_in = in_2.shape[1] if in_2.dim() > 1 else 1

    out = torch.empty_like(in_3)

    n_hw = H * W
    BLOCK_HW = triton.next_power_of_2(min(n_hw, 2048))
    if BLOCK_HW < 64:
        BLOCK_HW = 64

    grid = (B * C_out,)

    fused_se_kernel[grid](
        in_2, in_1, in_0, in_3, out,
        B, C_in, C_out, H, W,
        in_2.stride(0), in_2.stride(1) if in_2.dim() > 1 else 1,
        in_1.stride(0), in_1.stride(1),
        in_0.stride(0),
        in_3.stride(0), in_3.stride(1), in_3.stride(2), in_3.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_HW=BLOCK_HW,
    )

    return (out,)