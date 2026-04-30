import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.sigmoid()
    tmp_4 = in_2 * tmp_3
    tmp_5 = torch.nn.functional.hardtanh(tmp_4, 0.0, 6.0, False)
    return tmp_5


def replacement_args(in_0, in_1, in_2, in_3):
    device = in_2.device
    in_0_dev = in_0.to(device).contiguous()
    in_1_dev = in_1.to(device).contiguous().view(in_1.shape[0], in_1.shape[1])
    in_3_flat = in_3.contiguous().view(in_3.shape[0], in_3.shape[1])
    return (in_0_dev, in_1_dev, in_2, in_3_flat)


@triton.jit
def fused_se_kernel(
    in_0_ptr,     # bias [C_out]
    in_1_ptr,     # weight [C_out, C_in]
    in_2_ptr,     # x [B, C_out, H, W]
    in_3_ptr,     # x_se [B, C_in]
    out_ptr,      # output [B, C_out, H, W]
    C_in,
    C_out,
    HW,
    C_IN_BLOCK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Grid: (num_spatial_blocks, B * C_out)
    pid_spatial = tl.program_id(0)
    pid_bc = tl.program_id(1)

    b = pid_bc // C_out
    c_out = pid_bc % C_out

    # Step 1: Compute linear = dot(in_3[b,:], in_1[c_out,:]) + bias[c_out]
    bias = tl.load(in_0_ptr + c_out).to(tl.float32)

    c_in_offsets = tl.arange(0, C_IN_BLOCK)
    mask_cin = c_in_offsets < C_in

    in_3_vals = tl.load(in_3_ptr + b * C_in + c_in_offsets, mask=mask_cin, other=0.0)
    in_1_vals = tl.load(in_1_ptr + c_out * C_in + c_in_offsets, mask=mask_cin, other=0.0)

    dot = tl.sum(in_3_vals.to(tl.float32) * in_1_vals.to(tl.float32))
    linear_out = dot + bias

    # Step 2: Sigmoid (computed in fp32 for accuracy)
    sig = tl.sigmoid(linear_out)

    # Step 3: Load, multiply, clamp, store
    spatial_offset = pid_spatial * BLOCK_SIZE
    offsets = spatial_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < HW

    base_idx = pid_bc * HW
    in_2_vals = tl.load(in_2_ptr + base_idx + offsets, mask=mask, other=0.0)

    # Multiply in native precision (matching original PyTorch behavior)
    # and clamp
    result = in_2_vals.to(tl.float32) * sig
    result = tl.minimum(tl.maximum(result, 0.0), 6.0)

    tl.store(out_ptr + base_idx + offsets, result.to(in_2_vals.dtype), mask=mask)


@torch.fx.wrap
def fused_se_forward(in_0, in_1, in_2, in_3):
    B = in_2.shape[0]
    C_out = in_2.shape[1]
    H = in_2.shape[2]
    W = in_2.shape[3]
    C_in = in_3.shape[1]
    HW = H * W
    B_C_out = B * C_out

    C_IN_BLOCK = 1
    tmp_c = C_in
    while C_IN_BLOCK < tmp_c:
        C_IN_BLOCK *= 2

    out = torch.empty_like(in_2)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(HW, BLOCK_SIZE), B_C_out)

    fused_se_kernel[grid](
        in_0, in_1, in_2, in_3, out,
        C_in, C_out, HW,
        C_IN_BLOCK, BLOCK_SIZE,
        num_warps=4, num_stages=4,
    )

    return out


def replacement_func():
    return fused_se_forward