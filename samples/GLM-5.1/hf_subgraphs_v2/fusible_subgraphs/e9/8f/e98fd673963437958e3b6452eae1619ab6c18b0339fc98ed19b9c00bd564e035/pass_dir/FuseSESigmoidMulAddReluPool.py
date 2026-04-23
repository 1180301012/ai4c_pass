import torch
import triton
import triton.language as tl

# Pattern matching function - mirrors the exact computation from model.py
# IMPORTANT: exclude cleanup statements (tmp_x = None)
def pattern(in_0, in_1, in_2):
    tmp_0 = in_2.sigmoid()
    tmp_1 = tmp_0.view(1, -1, 1, 1)
    tmp_2 = tmp_1.expand_as(in_1)
    tmp_3 = in_1 * tmp_2
    tmp_3 += in_0
    tmp_4 = tmp_3
    tmp_5 = torch.nn.functional.relu(tmp_4, inplace=True)
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(tmp_5, 1)
    tmp_7 = tmp_6.flatten(1, -1)
    return (tmp_7,)

# Argument extraction - all three inputs needed
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# Triton kernel: fused sigmoid + broadcast_mul + add + relu + avg_pool + flatten
# Each program handles one or more channels, accumulating across spatial dims
@triton.jit
def fused_se_pool_kernel(
    in0_ptr,      # residual input: [N, C, H, W]
    in1_ptr,      # main input: [N, C, H, W]
    in2_ptr,      # sigmoid weights: [1, 1, C] or [C]
    out_ptr,      # output: [N, C]
    N, C, H, W,
    stride_in0_n, stride_in0_c, stride_in0_h, stride_in0_w,
    stride_in1_n, stride_in1_c, stride_in1_h, stride_in1_w,
    stride_in2_n, stride_in2_c, stride_in2_h, stride_in2_w,
    stride_out_n, stride_out_c,
    HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    c_start = pid * BLOCK_C
    c_offsets = c_start + tl.arange(0, BLOCK_C)
    c_mask = c_offsets < C

    # Load sigmoid weight for this channel block
    # in_2 shape is [1, 1, C] - we access element at channel c
    sig_ptrs = in2_ptr + c_offsets * stride_in2_c
    sig_vals = tl.load(sig_ptrs, mask=c_mask, other=0.0)
    # Compute sigmoid
    # Triton's sigmoid: 1/(1+exp(-x))
    sig_weights = tl.sigmoid(sig_vals)

    # Accumulate average pool across spatial dimensions
    # For each channel, sum relu(in1 * sig_weight + in0) over H*W, then divide by H*W
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)

    hw_offsets = tl.arange(0, HW)

    for n_idx in range(N):
        n_offset = n_idx * stride_in0_n  # same for in0 and in1 in typical case
        n_offset1 = n_idx * stride_in1_n

        for hw_idx in range(HW):
            h_idx = hw_idx // W
            w_idx = hw_idx % W
            spatial_offset_in0 = n_offset + c_offsets * stride_in0_c + h_idx * stride_in0_h + w_idx * stride_in0_w
            spatial_offset_in1 = n_offset1 + c_offsets * stride_in1_c + h_idx * stride_in1_h + w_idx * stride_in1_w

            # Load residual and main input for this spatial position
            in0_vals = tl.load(in0_ptr + spatial_offset_in0, mask=c_mask, other=0.0).to(tl.float32)
            in1_vals = tl.load(in1_ptr + spatial_offset_in1, mask=c_mask, other=0.0).to(tl.float32)

            # Fused: in1 * sigmoid_weight + in0, then relu
            fused = in1_vals * sig_weights + in0_vals
            relu_val = tl.where(fused > 0.0, fused, 0.0)

            # Accumulate for average pooling
            acc += relu_val

    # Average pool: divide by H*W
    pool_result = acc / HW

    # Store output: [N, C]
    for n_idx in range(N):
        out_ptrs = out_ptr + n_idx * stride_out_n + c_offsets * stride_out_c
        tl.store(out_ptrs, pool_result.to(out_ptr.dtype.element_ty), mask=c_mask)


@torch.fx.wrap
def fused_se_pool(in_0, in_1, in_2):
    # Get dimensions
    # in_0, in_1: [N, C, H, W]
    # in_2: [1, 1, C]
    N, C, H, W = in_0.shape
    HW = H * W

    # Output shape after adaptive_avg_pool2d(1) + flatten: [N, C]
    out = torch.empty((N, C), dtype=in_0.dtype, device=in_0.device)

    BLOCK_C = 32  # Process 32 channels per program
    num_programs = (C + BLOCK_C - 1) // BLOCK_C

    grid = (num_programs,)

    fused_se_pool_kernel[grid](
        in_0, in_1, in_2, out,
        N, C, H, W,
        in_0.stride(0), in_0.stride(1), in_0.stride(2), in_0.stride(3),
        in_1.stride(0), in_1.stride(1), in_1.stride(2), in_1.stride(3),
        in_2.stride(0), in_2.stride(1), in_2.stride(2), in_2.stride(3),
        out.stride(0), out.stride(1),
        HW=HW,
        BLOCK_C=BLOCK_C,
    )

    return out

def replacement_func():
    return fused_se_pool