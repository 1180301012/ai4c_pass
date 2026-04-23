import torch
import triton
import triton.language as tl


# Pattern matching function for 16-head, 256-seq config
def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    tmp_1 = in_3 * in_1
    tmp_2 = in_3[Ellipsis, slice(1, None, 2)]
    tmp_3 = -tmp_2
    tmp_4 = in_3[Ellipsis, slice(None, None, 2)]
    tmp_5 = torch.stack([tmp_3, tmp_4], -1)
    tmp_6 = tmp_5.reshape((1, 16, 256, 64))
    tmp_7 = tmp_6 * in_5
    tmp_8 = tmp_1 + tmp_7
    tmp_9 = torch.cat([in_2, tmp_8], dim=2)
    tmp_10 = tmp_9.type_as(in_6)
    
    tmp_11 = in_4[slice(None, None, None), slice(None, None, None), slice(None, 1, None), slice(None, None, None)]
    tmp_12 = in_4[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)]
    tmp_13 = in_0.tensor_split(2, -1)
    tmp_14 = tmp_13[0]
    tmp_15 = tmp_13[1]
    tmp_16 = tmp_12 * tmp_15
    tmp_17 = tmp_12[Ellipsis, slice(1, None, 2)]
    tmp_18 = -tmp_17
    tmp_19 = tmp_12[Ellipsis, slice(None, None, 2)]
    tmp_20 = torch.stack([tmp_18, tmp_19], -1)
    tmp_21 = tmp_20.reshape((1, 16, 256, 64))
    tmp_22 = tmp_21 * tmp_14
    tmp_23 = tmp_16 + tmp_22
    tmp_24 = torch.cat([tmp_11, tmp_23], dim=2)
    tmp_25 = tmp_24.type_as(in_6)
    
    return (tmp_25, tmp_10)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6)


@triton.jit
def rope_q_kernel(
    in_3_ptr, in_1_ptr, in_5_ptr, in_2_ptr, out_ptr,
    B, H, S, D,
    stride_in3_b, stride_in3_h, stride_in3_s, stride_in3_d,
    stride_in1_s, stride_in1_d,
    stride_in5_s, stride_in5_d,
    stride_in2_b, stride_in2_h, stride_in2_d,
    stride_out_b, stride_out_h, stride_out_s, stride_out_d,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)
    
    offs_s = pid_s * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_N)
    
    mask_s = offs_s < (S + 1)
    mask_d = offs_d < D
    mask = mask_s[:, None] & mask_d[None, :]
    
    off_b = pid_b * stride_in3_b
    off_h = pid_h * stride_in3_h
    
    is_first = offs_s == 0
    
    in_2_off = off_b + pid_h * stride_in2_h + 0 * stride_out_s + offs_d[None, :] * stride_in2_d
    in_2_val = tl.load(in_2_ptr + in_2_off, mask=mask_d[None, :], other=0.0)
    
    s_in3 = offs_s - 1
    
    in_3_off = off_b + off_h + s_in3[:, None] * stride_in3_s + offs_d[None, :] * stride_in3_d
    in_3_val = tl.load(in_3_ptr + in_3_off, mask=mask & (offs_s > 0)[:, None], other=0.0)
    
    in_1_off = s_in3[:, None] * stride_in1_s + offs_d[None, :] * stride_in1_d
    in_1_val = tl.load(in_1_ptr + in_1_off, mask=mask & (offs_s > 0)[:, None], other=0.0)
    tmp_1 = in_3_val * in_1_val
    
    src_d = tl.where(offs_d % 2 == 0, offs_d + 1, offs_d - 1)
    src_d = tl.minimum(src_d, D - 1)
    src_off = off_b + off_h + s_in3[:, None] * stride_in3_s + src_d[None, :] * stride_in3_d
    src_val = tl.load(in_3_ptr + src_off, mask=mask & (offs_s > 0)[:, None], other=0.0)
    interleaved = tl.where(offs_d % 2 == 0, -src_val, src_val)
    
    in_5_off = s_in3[:, None] * stride_in5_s + offs_d[None, :] * stride_in5_d
    in_5_val = tl.load(in_5_ptr + in_5_off, mask=mask & (offs_s > 0)[:, None], other=0.0)
    tmp_7 = interleaved * in_5_val
    
    tmp_8 = tmp_1 + tmp_7
    rope_result = tl.where(is_first[:, None], in_2_val, tmp_8)
    
    out_off = off_b + pid_h * stride_out_h + offs_s[:, None] * stride_out_s + offs_d[None, :] * stride_out_d
    tl.store(out_ptr + out_off, rope_result, mask=mask)


@triton.jit
def rope_k_kernel(
    in_0_ptr, in_4_ptr, out_ptr,
    B, H, S, D,
    stride_in0_s, stride_in0_d,
    stride_in4_b, stride_in4_h, stride_in4_s, stride_in4_d,
    stride_out_b, stride_out_h, stride_out_s, stride_out_d,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)
    
    offs_s = pid_s * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_N)
    
    mask_s = offs_s < (S + 1)
    mask_d = offs_d < D
    mask = mask_s[:, None] & mask_d[None, :]
    
    off_b = pid_b * stride_in4_b
    off_h = pid_h * stride_in4_h
    
    is_first = offs_s == 0
    
    in_4_first_off = off_b + off_h + 0 * stride_in4_s + offs_d[None, :] * stride_in4_d
    in_4_first_val = tl.load(in_4_ptr + in_4_first_off, mask=mask_d[None, :], other=0.0)
    
    s_k = offs_s
    s_in0 = offs_s - 1
    
    in_4_off = off_b + off_h + s_k[:, None] * stride_in4_s + offs_d[None, :] * stride_in4_d
    in_4_val = tl.load(in_4_ptr + in_4_off, mask=mask & (offs_s > 0)[:, None], other=0.0)
    
    in_0_second_off = s_in0[:, None] * stride_in0_s + (D + offs_d)[None, :] * stride_in0_d
    in_0_second_val = tl.load(in_0_ptr + in_0_second_off, mask=mask & (offs_s > 0)[:, None], other=0.0)
    tmp_16 = in_4_val * in_0_second_val
    
    src_d = tl.where(offs_d % 2 == 0, offs_d + 1, offs_d - 1)
    src_d = tl.minimum(src_d, D - 1)
    src_off = off_b + off_h + s_k[:, None] * stride_in4_s + src_d[None, :] * stride_in4_d
    src_val = tl.load(in_4_ptr + src_off, mask=mask & (offs_s > 0)[:, None], other=0.0)
    interleaved = tl.where(offs_d % 2 == 0, -src_val, src_val)
    
    in_0_first_off = s_in0[:, None] * stride_in0_s + offs_d[None, :] * stride_in0_d
    in_0_first_val = tl.load(in_0_ptr + in_0_first_off, mask=mask & (offs_s > 0)[:, None], other=0.0)
    tmp_22 = interleaved * in_0_first_val
    
    tmp_23 = tmp_16 + tmp_22
    k_result = tl.where(is_first[:, None], in_4_first_val, tmp_23)
    
    out_off = off_b + pid_h * stride_out_h + offs_s[:, None] * stride_out_s + offs_d[None, :] * stride_out_d
    tl.store(out_ptr + out_off, k_result, mask=mask)


@torch.fx.wrap
def rope_kernel_wrapper(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    B, H, S, D = in_3.shape
    
    out_1_shape = (B, H, S + in_2.shape[2], D)
    out_1 = torch.empty(out_1_shape, dtype=in_6.dtype, device=in_6.device)
    
    out_0_shape = (B, H, S + 1, D)
    out_0 = torch.empty(out_0_shape, dtype=in_6.dtype, device=in_6.device)
    
    BLOCK_M = 16
    BLOCK_N = 64
    
    grid_q = (B, H, (S + in_2.shape[2] + BLOCK_M - 1) // BLOCK_M)
    rope_q_kernel[grid_q](
        in_3, in_1, in_5, in_2, out_1,
        B, H, S, D,
        in_3.stride(0), in_3.stride(1), in_3.stride(2), in_3.stride(3),
        in_1.stride(0), in_1.stride(1),
        in_5.stride(0), in_5.stride(1),
        in_2.stride(0), in_2.stride(1), in_2.stride(3),
        out_1.stride(0), out_1.stride(1), out_1.stride(2), out_1.stride(3),
        BLOCK_M, BLOCK_N,
    )
    
    grid_k = (B, H, (S + 1 + BLOCK_M - 1) // BLOCK_M)
    rope_k_kernel[grid_k](
        in_0, in_4, out_0,
        B, H, S, D,
        in_0.stride(0), in_0.stride(1),
        in_4.stride(0), in_4.stride(1), in_4.stride(2), in_4.stride(3),
        out_0.stride(0), out_0.stride(1), out_0.stride(2), out_0.stride(3),
        BLOCK_M, BLOCK_N,
    )
    
    return (out_0, out_1)


def replacement_func():
    return rope_kernel_wrapper