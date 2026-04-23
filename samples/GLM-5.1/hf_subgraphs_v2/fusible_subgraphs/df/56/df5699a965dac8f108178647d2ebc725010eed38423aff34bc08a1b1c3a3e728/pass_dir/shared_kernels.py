import torch
import triton
import triton.language as tl


# Path 1 shuffle kernel: channel shuffle for (in_2, in_4) -> (chunk0, chunk1)
# Groups=2, C_per_group=20, output chunks each have 20 channels
@triton.jit
def shuffle_path1_kernel(
    src1_ptr, src2_ptr, out_chunk0_ptr, out_chunk1_ptr,
    B,
    C_per_group: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)  # 0..C_per_group-1 = 0..19

    HW = H * W

    # Source info for chunk[0]:
    # shuffled_c = pid_c, group = pid_c % 2, within_group = pid_c // 2
    group0 = pid_c % 2
    src_c0 = pid_c // 2

    # Source info for chunk[1]:
    # shuffled_c = C_per_group + pid_c
    shuffled_c1 = C_per_group + pid_c
    group1 = shuffled_c1 % 2
    src_c1 = shuffled_c1 // 2

    for hw_start in range(0, HW, BLOCK_HW):
        hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
        hw_mask = hw_offsets < HW

        # Load for chunk[0]
        src0_off = pid_b * (C_per_group * HW) + src_c0 * HW + hw_offsets
        if group0 == 0:
            val0 = tl.load(src1_ptr + src0_off, mask=hw_mask, other=0.0)
        else:
            val0 = tl.load(src2_ptr + src0_off, mask=hw_mask, other=0.0)

        # Load for chunk[1]
        src1_off = pid_b * (C_per_group * HW) + src_c1 * HW + hw_offsets
        if group1 == 0:
            val1 = tl.load(src1_ptr + src1_off, mask=hw_mask, other=0.0)
        else:
            val1 = tl.load(src2_ptr + src1_off, mask=hw_mask, other=0.0)

        # Store to chunk[0]
        out0_off = pid_b * (C_per_group * HW) + pid_c * HW + hw_offsets
        tl.store(out_chunk0_ptr + out0_off, val0, mask=hw_mask)

        # Store to chunk[1]
        out1_off = pid_b * (C_per_group * HW) + pid_c * HW + hw_offsets
        tl.store(out_chunk1_ptr + out1_off, val1, mask=hw_mask)


# SE attention + path 2 shuffle kernel
# Computes: conv2d(in_6, in_1, in_0) + sigmoid + broadcast mul with in_5
# Then channel shuffle for (in_3, tmp_4) -> (chunk0, chunk1)
# Groups=2, each group has C_out=40 channels, output chunks each have 40 channels
@triton.jit
def fused_se_shuffle_path2_kernel(
    input_ptr, weight_ptr, bias_ptr, in3_ptr, in5_ptr,
    out_chunk0_ptr, out_chunk1_ptr,
    B,
    C_in: tl.constexpr,
    C_out: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_co = tl.program_id(1)  # 0..C_out-1 = 0..39

    # Conv2d: vectorized dot product over C_in channels
    ci_offsets = tl.arange(0, C_in)
    input_vals = tl.load(input_ptr + pid_b * C_in + ci_offsets).to(tl.float32)
    weight_vals = tl.load(weight_ptr + pid_co * C_in + ci_offsets).to(tl.float32)
    acc = tl.sum(input_vals * weight_vals)

    # Add bias and apply sigmoid
    bias_val = tl.load(bias_ptr + pid_co).to(tl.float32)
    conv_out = acc + bias_val
    sig_val = tl.sigmoid(conv_out)

    HW = H * W
    HALF = C_out // 2  # = 20

    for hw_start in range(0, HW, BLOCK_HW):
        hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
        hw_mask = hw_offsets < HW

        # Load in_3[b, pid_co, h, w]
        in3_off = pid_b * C_out * HW + pid_co * HW + hw_offsets
        in3_val = tl.load(in3_ptr + in3_off, mask=hw_mask, other=0.0)

        # Load in_5[b, pid_co, h, w] and compute tmp_4 = in_5 * sigmoid(conv)
        in5_off = pid_b * C_out * HW + pid_co * HW + hw_offsets
        in5_val = tl.load(in5_ptr + in5_off, mask=hw_mask, other=0.0)
        tmp4_val = in5_val * sig_val

        # Write to appropriate output chunk
        if pid_co < HALF:
            # Write to chunk[0]
            out_c_in3 = 2 * pid_co      # even channel position
            out_c_tmp4 = 2 * pid_co + 1  # odd channel position

            out_in3_off = pid_b * C_out * HW + out_c_in3 * HW + hw_offsets
            out_tmp4_off = pid_b * C_out * HW + out_c_tmp4 * HW + hw_offsets

            tl.store(out_chunk0_ptr + out_in3_off, in3_val, mask=hw_mask)
            tl.store(out_chunk0_ptr + out_tmp4_off, tmp4_val, mask=hw_mask)
        else:
            # Write to chunk[1]
            k = pid_co - HALF
            out_c_in3 = 2 * k      # even channel position
            out_c_tmp4 = 2 * k + 1  # odd channel position

            out_in3_off = pid_b * C_out * HW + out_c_in3 * HW + hw_offsets
            out_tmp4_off = pid_b * C_out * HW + out_c_tmp4 * HW + hw_offsets

            tl.store(out_chunk1_ptr + out_in3_off, in3_val, mask=hw_mask)
            tl.store(out_chunk1_ptr + out_tmp4_off, tmp4_val, mask=hw_mask)


@torch.fx.wrap
def fused_model_dispatch(in_0, in_1, in_2, in_3, in_4, in_5, in_6, route=""):
    B = in_2.shape[0]
    dtype = in_2.dtype
    device = in_2.device

    # Allocate outputs for path 1 shuffle: [B, 20, 64, 48]
    out_chunk0_p1 = torch.empty(B, 20, 64, 48, dtype=dtype, device=device)
    out_chunk1_p1 = torch.empty(B, 20, 64, 48, dtype=dtype, device=device)

    # Allocate outputs for path 2 shuffle + SE attention: [B, 40, 32, 24]
    out_chunk0_p2 = torch.empty(B, 40, 32, 24, dtype=dtype, device=device)
    out_chunk1_p2 = torch.empty(B, 40, 32, 24, dtype=dtype, device=device)

    # Launch path 1 shuffle kernel
    grid1 = (B, 20)
    shuffle_path1_kernel[grid1](
        src1_ptr=in_2, src2_ptr=in_4,
        out_chunk0_ptr=out_chunk0_p1, out_chunk1_ptr=out_chunk1_p1,
        B=B, C_per_group=20, H=64, W=48,
        BLOCK_HW=512,
    )

    # Launch SE attention + path 2 shuffle kernel
    grid2 = (B, 40)
    fused_se_shuffle_path2_kernel[grid2](
        input_ptr=in_6, weight_ptr=in_1, bias_ptr=in_0,
        in3_ptr=in_3, in5_ptr=in_5,
        out_chunk0_ptr=out_chunk0_p2, out_chunk1_ptr=out_chunk1_p2,
        B=B, C_in=10, C_out=40, H=32, W=24,
        BLOCK_HW=256,
    )

    # Return in same order as model: (tmp_16, tmp_19, tmp_17, tmp_20)
    # tmp_16 = chunk[0] of shuffled path 1 = out_chunk0_p1
    # tmp_19 = chunk[0] of shuffled path 2 = out_chunk0_p2
    # tmp_17 = chunk[1] of shuffled path 1 = out_chunk1_p1
    # tmp_20 = chunk[1] of shuffled path 2 = out_chunk1_p2
    return (out_chunk0_p1, out_chunk0_p2, out_chunk1_p1, out_chunk1_p2)