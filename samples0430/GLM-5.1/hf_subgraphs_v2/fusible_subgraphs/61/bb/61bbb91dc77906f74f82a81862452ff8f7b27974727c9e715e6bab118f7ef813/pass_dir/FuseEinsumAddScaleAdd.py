import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4):
    einsum = torch.functional.einsum('bchj,bhwj->bchw', in_4, in_1)
    add_result = in_3 + einsum
    mul_result = add_result * in_0
    add_result2 = mul_result + in_2
    result = add_result2.contiguous()
    return result


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


@triton.jit
def fused_einsum_add_scale_add_kernel(
    in_4_ptr, in_1_ptr, in_3_ptr, in_2_ptr, out_ptr,
    in_0_val,
    B, C, H, W, J,
    stride_in4_b, stride_in4_c, stride_in4_h, stride_in4_j,
    stride_in1_b, stride_in1_h, stride_in1_w, stride_in1_j,
    stride_in3_b, stride_in3_c, stride_in3_h, stride_in3_w,
    stride_in2_b, stride_in2_c, stride_in2_h, stride_in2_w,
    stride_out_b, stride_out_c, stride_out_h, stride_out_w,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    DTYPE: tl.constexpr,  # 0=fp32, 1=fp16, 2=bf16
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    b = pid_bh // H
    h = pid_bh - b * H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, J, BLOCK_K):
        offs_k_cur = k_start + offs_k

        # Load A tile: in_4[b, offs_m, h, offs_k_cur] -> [BLOCK_M, BLOCK_K]
        # A[c,j] = in_4[b, c, h, j]
        a_ptrs = in_4_ptr + b * stride_in4_b + h * stride_in4_h + offs_m[:, None] * stride_in4_c + offs_k_cur[None, :] * stride_in4_j
        a_mask = (offs_m[:, None] < C) & (offs_k_cur[None, :] < J)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B^T tile: [BLOCK_K, BLOCK_N]
        # The einsum: result[b,c,h,w] = sum_j in_4[b,c,h,j] * in_1[b,h,w,j]
        # For matmul view: result[c,w] = sum_j A[c,j] * B[w,j] = A @ B^T
        # B^T[j,w] = in_1[b,h,w,j]
        # So load in_1[b, h, w, j] with j as row, w as column
        b_t_ptrs = in_1_ptr + b * stride_in1_b + h * stride_in1_h + offs_k_cur[:, None] * stride_in1_j + offs_n[None, :] * stride_in1_w
        b_t_mask = (offs_k_cur[:, None] < J) & (offs_n[None, :] < W)
        b_t = tl.load(b_t_ptrs, mask=b_t_mask, other=0.0)

        acc += tl.dot(a, b_t)

    # After matmul, convert acc to target dtype to match PyTorch einsum output behavior
    # Then do element-wise operations in target dtype to match PyTorch precision
    if DTYPE == 0:  # fp32
        acc_t = acc
        in0_t = in_0_val
    elif DTYPE == 1:  # fp16
        acc_t = acc.to(tl.float16)
        in0_t = in_0_val.to(tl.float16)
    else:  # bf16
        acc_t = acc.to(tl.bfloat16)
        in0_t = in_0_val.to(tl.bfloat16)

    # Load in_3 (in target dtype)
    in3_ptrs = in_3_ptr + b * stride_in3_b + h * stride_in3_h + offs_m[:, None] * stride_in3_c + offs_n[None, :] * stride_in3_w
    in3_mask = (offs_m[:, None] < C) & (offs_n[None, :] < W)
    in3 = tl.load(in3_ptrs, mask=in3_mask, other=0.0)

    # Load in_2 (in target dtype)
    in2_ptrs = in_2_ptr + b * stride_in2_b + h * stride_in2_h + offs_m[:, None] * stride_in2_c + offs_n[None, :] * stride_in2_w
    in2_mask = (offs_m[:, None] < C) & (offs_n[None, :] < W)
    in2 = tl.load(in2_ptrs, mask=in2_mask, other=0.0)

    # Compute: (in_3 + einsum) * in_0 + in_2 in target dtype
    result = (in3 + acc_t) * in0_t + in2

    # Store result
    out_ptrs = out_ptr + b * stride_out_b + h * stride_out_h + offs_m[:, None] * stride_out_c + offs_n[None, :] * stride_out_w
    out_mask = (offs_m[:, None] < C) & (offs_n[None, :] < W)
    tl.store(out_ptrs, result, mask=out_mask)


@torch.fx.wrap
def fused_einsum_add_scale_add(in_0, in_1, in_2, in_3, in_4):
    B, C, H, J = in_4.shape
    W = in_1.shape[2]

    dtype = in_4.dtype

    # Determine dtype flag
    if dtype == torch.float32:
        dtype_flag = 0
    elif dtype == torch.float16:
        dtype_flag = 1
    elif dtype == torch.bfloat16:
        dtype_flag = 2
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # Allocate output in target dtype
    out = torch.empty(B, C, H, W, dtype=dtype, device=in_4.device)

    in_0_val = in_0.item() if in_0.numel() == 1 else in_0

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64

    grid = (B * H, triton.cdiv(C, BLOCK_M), triton.cdiv(W, BLOCK_N))

    fused_einsum_add_scale_add_kernel[grid](
        in_4_ptr=in_4,
        in_1_ptr=in_1,
        in_3_ptr=in_3,
        in_2_ptr=in_2,
        out_ptr=out,
        in_0_val=in_0_val,
        B=B, C=C, H=H, W=W, J=J,
        stride_in4_b=in_4.stride(0), stride_in4_c=in_4.stride(1),
        stride_in4_h=in_4.stride(2), stride_in4_j=in_4.stride(3),
        stride_in1_b=in_1.stride(0), stride_in1_h=in_1.stride(1),
        stride_in1_w=in_1.stride(2), stride_in1_j=in_1.stride(3),
        stride_in3_b=in_3.stride(0), stride_in3_c=in_3.stride(1),
        stride_in3_h=in_3.stride(2), stride_in3_w=in_3.stride(3),
        stride_in2_b=in_2.stride(0), stride_in2_c=in_2.stride(1),
        stride_in2_h=in_2.stride(2), stride_in2_w=in_2.stride(3),
        stride_out_b=out.stride(0), stride_out_c=out.stride(1),
        stride_out_h=out.stride(2), stride_out_w=out.stride(3),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        DTYPE=dtype_flag,
    )

    return out.contiguous()


def replacement_func():
    return fused_einsum_add_scale_add