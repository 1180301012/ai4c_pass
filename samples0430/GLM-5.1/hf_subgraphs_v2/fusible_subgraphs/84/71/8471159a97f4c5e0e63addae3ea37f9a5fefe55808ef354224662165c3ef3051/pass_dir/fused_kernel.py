import torch
import triton
import triton.language as tl


@triton.jit
def _fused_linear_sigmoid_mul_kernel_bf16(
    in_0_ptr, in_1_ptr, in_2_ptr, in_3_ptr, out_ptr,
    B, C, H, W,
    stride_in0,
    stride_in1_c, stride_in1_k,
    stride_in2_b, stride_in2_k,
    stride_in3_b, stride_in3_c, stride_in3_h, stride_in3_w,
    stride_out_b, stride_out_c, stride_out_h, stride_out_w,
    BLOCK_HW: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    bc_id = tl.program_id(0)
    hw_id = tl.program_id(1)

    b = bc_id // C
    c = bc_id % C

    # Step 1: Compute weight = sigmoid(in_2[b] @ in_1[c] + in_0[c])
    bias = tl.load(in_0_ptr + c * stride_in0).to(tl.float32)

    # Compute dot product in fp32: in_2[b, :] @ in_1[c, :]
    k_offsets = tl.arange(0, BLOCK_K)
    in2_vals = tl.load(in_2_ptr + b * stride_in2_b + k_offsets * stride_in2_k).to(tl.float32)
    in1_vals = tl.load(in_1_ptr + c * stride_in1_c + k_offsets * stride_in1_k).to(tl.float32)
    acc = in2_vals * in1_vals

    linear_val = tl.sum(acc) + bias
    weight = tl.sigmoid(linear_val)

    # Step 2: Multiply in_3[b, c, h, w] * weight for a block of HW
    hw_start = hw_id * BLOCK_HW
    hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
    h_off = hw_offsets // W
    w_off = hw_offsets % W
    hw_mask = hw_offsets < H * W

    in3_ptrs = in_3_ptr + b * stride_in3_b + c * stride_in3_c + h_off * stride_in3_h + w_off * stride_in3_w
    in3_vals = tl.load(in3_ptrs, mask=hw_mask, other=0.0).to(tl.float32)

    out_vals = in3_vals * weight

    out_ptrs = out_ptr + b * stride_out_b + c * stride_out_c + h_off * stride_out_h + w_off * stride_out_w
    tl.store(out_ptrs, out_vals.to(tl.bfloat16), mask=hw_mask)


@triton.jit
def _fused_linear_sigmoid_mul_kernel_f16(
    in_0_ptr, in_1_ptr, in_2_ptr, in_3_ptr, out_ptr,
    B, C, H, W,
    stride_in0,
    stride_in1_c, stride_in1_k,
    stride_in2_b, stride_in2_k,
    stride_in3_b, stride_in3_c, stride_in3_h, stride_in3_w,
    stride_out_b, stride_out_c, stride_out_h, stride_out_w,
    BLOCK_HW: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    bc_id = tl.program_id(0)
    hw_id = tl.program_id(1)

    b = bc_id // C
    c = bc_id % C

    bias = tl.load(in_0_ptr + c * stride_in0).to(tl.float32)

    k_offsets = tl.arange(0, BLOCK_K)
    in2_vals = tl.load(in_2_ptr + b * stride_in2_b + k_offsets * stride_in2_k).to(tl.float32)
    in1_vals = tl.load(in_1_ptr + c * stride_in1_c + k_offsets * stride_in1_k).to(tl.float32)
    acc = in2_vals * in1_vals

    linear_val = tl.sum(acc) + bias
    weight = tl.sigmoid(linear_val)

    hw_start = hw_id * BLOCK_HW
    hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
    h_off = hw_offsets // W
    w_off = hw_offsets % W
    hw_mask = hw_offsets < H * W

    in3_ptrs = in_3_ptr + b * stride_in3_b + c * stride_in3_c + h_off * stride_in3_h + w_off * stride_in3_w
    in3_vals = tl.load(in3_ptrs, mask=hw_mask, other=0.0).to(tl.float32)

    out_vals = in3_vals * weight

    out_ptrs = out_ptr + b * stride_out_b + c * stride_out_c + h_off * stride_out_h + w_off * stride_out_w
    tl.store(out_ptrs, out_vals.to(tl.float16), mask=hw_mask)


@triton.jit
def _fused_linear_sigmoid_mul_kernel_f32(
    in_0_ptr, in_1_ptr, in_2_ptr, in_3_ptr, out_ptr,
    B, C, H, W,
    stride_in0,
    stride_in1_c, stride_in1_k,
    stride_in2_b, stride_in2_k,
    stride_in3_b, stride_in3_c, stride_in3_h, stride_in3_w,
    stride_out_b, stride_out_c, stride_out_h, stride_out_w,
    BLOCK_HW: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    bc_id = tl.program_id(0)
    hw_id = tl.program_id(1)

    b = bc_id // C
    c = bc_id % C

    bias = tl.load(in_0_ptr + c * stride_in0)

    k_offsets = tl.arange(0, BLOCK_K)
    in2_vals = tl.load(in_2_ptr + b * stride_in2_b + k_offsets * stride_in2_k)
    in1_vals = tl.load(in_1_ptr + c * stride_in1_c + k_offsets * stride_in1_k)
    acc = in2_vals * in1_vals

    linear_val = tl.sum(acc) + bias
    weight = tl.sigmoid(linear_val)

    hw_start = hw_id * BLOCK_HW
    hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
    h_off = hw_offsets // W
    w_off = hw_offsets % W
    hw_mask = hw_offsets < H * W

    in3_ptrs = in_3_ptr + b * stride_in3_b + c * stride_in3_c + h_off * stride_in3_h + w_off * stride_in3_w
    in3_vals = tl.load(in3_ptrs, mask=hw_mask, other=0.0)

    out_vals = in3_vals * weight

    out_ptrs = out_ptr + b * stride_out_b + c * stride_out_c + h_off * stride_out_h + w_off * stride_out_w
    tl.store(out_ptrs, out_vals, mask=hw_mask)


@torch.fx.wrap
def _fused_linear_sigmoid_mul_impl(in_0, in_1, in_2, in_3):
    B = in_2.shape[0]
    C = in_1.shape[0]
    H = in_3.shape[2]
    W = in_3.shape[3]

    out = torch.empty_like(in_3)

    stride_in0 = in_0.stride(0)
    stride_in1_c = in_1.stride(0)
    stride_in1_k = in_1.stride(1)
    stride_in2_b = in_2.stride(0)
    stride_in2_k = in_2.stride(1)
    stride_in3_b = in_3.stride(0)
    stride_in3_c = in_3.stride(1)
    stride_in3_h = in_3.stride(2)
    stride_in3_w = in_3.stride(3)
    stride_out_b = out.stride(0)
    stride_out_c = out.stride(1)
    stride_out_h = out.stride(2)
    stride_out_w = out.stride(3)

    BLOCK_HW = 256
    BLOCK_K = 8
    hw_total = H * W
    grid = (B * C, triton.cdiv(hw_total, BLOCK_HW))

    if in_3.dtype == torch.bfloat16:
        _fused_linear_sigmoid_mul_kernel_bf16[grid](
            in_0_ptr=in_0, in_1_ptr=in_1, in_2_ptr=in_2, in_3_ptr=in_3, out_ptr=out,
            B=B, C=C, H=H, W=W,
            stride_in0=stride_in0,
            stride_in1_c=stride_in1_c, stride_in1_k=stride_in1_k,
            stride_in2_b=stride_in2_b, stride_in2_k=stride_in2_k,
            stride_in3_b=stride_in3_b, stride_in3_c=stride_in3_c, stride_in3_h=stride_in3_h, stride_in3_w=stride_in3_w,
            stride_out_b=stride_out_b, stride_out_c=stride_out_c, stride_out_h=stride_out_h, stride_out_w=stride_out_w,
            BLOCK_HW=BLOCK_HW,
            BLOCK_K=BLOCK_K,
        )
    elif in_3.dtype == torch.float16:
        _fused_linear_sigmoid_mul_kernel_f16[grid](
            in_0_ptr=in_0, in_1_ptr=in_1, in_2_ptr=in_2, in_3_ptr=in_3, out_ptr=out,
            B=B, C=C, H=H, W=W,
            stride_in0=stride_in0,
            stride_in1_c=stride_in1_c, stride_in1_k=stride_in1_k,
            stride_in2_b=stride_in2_b, stride_in2_k=stride_in2_k,
            stride_in3_b=stride_in3_b, stride_in3_c=stride_in3_c, stride_in3_h=stride_in3_h, stride_in3_w=stride_in3_w,
            stride_out_b=stride_out_b, stride_out_c=stride_out_c, stride_out_h=stride_out_h, stride_out_w=stride_out_w,
            BLOCK_HW=BLOCK_HW,
            BLOCK_K=BLOCK_K,
        )
    else:
        _fused_linear_sigmoid_mul_kernel_f32[grid](
            in_0_ptr=in_0, in_1_ptr=in_1, in_2_ptr=in_2, in_3_ptr=in_3, out_ptr=out,
            B=B, C=C, H=H, W=W,
            stride_in0=stride_in0,
            stride_in1_c=stride_in1_c, stride_in1_k=stride_in1_k,
            stride_in2_b=stride_in2_b, stride_in2_k=stride_in2_k,
            stride_in3_b=stride_in3_b, stride_in3_c=stride_in3_c, stride_in3_h=stride_in3_h, stride_in3_w=stride_in3_w,
            stride_out_b=stride_out_b, stride_out_c=stride_out_c, stride_out_h=stride_out_h, stride_out_w=stride_out_w,
            BLOCK_HW=BLOCK_HW,
            BLOCK_K=BLOCK_K,
        )

    return out


@torch.fx.wrap
def fused_linear_sigmoid_mul_dispatch(in_0, in_1, in_2, in_3, route=""):
    # Shared dispatch wrapper for all routes
    if route == "route_b1":
        return _fused_linear_sigmoid_mul_impl(in_0, in_1, in_2, in_3)
    elif route == "route_b32":
        return _fused_linear_sigmoid_mul_impl(in_0, in_1, in_2, in_3)
    elif route == "route_b128":
        return _fused_linear_sigmoid_mul_impl(in_0, in_1, in_2, in_3)
    else:
        return _fused_linear_sigmoid_mul_impl(in_0, in_1, in_2, in_3)