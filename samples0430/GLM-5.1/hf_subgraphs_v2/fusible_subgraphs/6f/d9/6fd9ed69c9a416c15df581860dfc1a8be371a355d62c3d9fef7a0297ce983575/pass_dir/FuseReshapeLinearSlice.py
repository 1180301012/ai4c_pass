import torch
import triton
import triton.language as tl


def pattern(in_4, in_3, in_2):
    tmp_9 = in_4.reshape(300, -1, 256)
    linear_1 = torch.nn.functional.linear(tmp_9, in_3, in_2)
    tmp_11 = linear_1[Ellipsis, slice(None, 256, None)]
    tmp_12 = linear_1[Ellipsis, slice(-256, None, None)]
    return tmp_11, tmp_12


def replacement_args(in_4, in_3, in_2):
    x = in_4.reshape(300, -1, 256)
    M = x.shape[0]
    N_half = in_3.shape[0] // 2
    weight_first = in_3[:N_half]
    weight_second = in_3[N_half:]
    bias_first = in_2[:N_half]
    bias_second = in_2[N_half:]
    # Flatten x to 2D for matmul
    x_flat = x.reshape(M, x.shape[-1])
    return (x_flat, weight_first, bias_first, weight_second, bias_second)


@triton.jit
def fused_linear_slice_both_3d_kernel(
    x_ptr, weight1_ptr, bias1_ptr, weight2_ptr, bias2_ptr,
    out_first_ptr, out_second_ptr,
    M, K, N_half,
    stride_xm, stride_xk,
    stride_w1n, stride_w1k,
    stride_b1,
    stride_w2n, stride_w2k,
    stride_b2,
    stride_out1_m, stride_out1_n,
    stride_out2_m, stride_out2_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    DTYPE_OUT: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N_half, BLOCK_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Two accumulators for two halves
    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load x tile: [BLOCK_M, BLOCK_K]
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load weight1 tile in transposed layout: [BLOCK_K, BLOCK_N]
        w1_ptrs = weight1_ptr + offs_k[:, None] * stride_w1k + offs_n[None, :] * stride_w1n
        w1_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N_half)
        w1 = tl.load(w1_ptrs, mask=w1_mask, other=0.0)

        # Load weight2 tile in transposed layout: [BLOCK_K, BLOCK_N]
        w2_ptrs = weight2_ptr + offs_k[:, None] * stride_w2k + offs_n[None, :] * stride_w2n
        w2_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N_half)
        w2 = tl.load(w2_ptrs, mask=w2_mask, other=0.0)

        # Compute both dot products - x is reused for both!
        acc1 += tl.dot(x, w1)
        acc2 += tl.dot(x, w2)

    # Add bias for both halves
    b1_ptrs = bias1_ptr + offs_n * stride_b1
    b1_mask = offs_n < N_half
    b1 = tl.load(b1_ptrs, mask=b1_mask, other=0.0).to(tl.float32)
    acc1 += b1[None, :]

    b2_ptrs = bias2_ptr + offs_n * stride_b2
    b2_mask = offs_n < N_half
    b2 = tl.load(b2_ptrs, mask=b2_mask, other=0.0).to(tl.float32)
    acc2 += b2[None, :]

    # Convert to output dtype and store
    out1_ptrs = out_first_ptr + offs_m[:, None] * stride_out1_m + offs_n[None, :] * stride_out1_n
    out1_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N_half)
    tl.store(out1_ptrs, acc1.to(DTYPE_OUT), mask=out1_mask)

    out2_ptrs = out_second_ptr + offs_m[:, None] * stride_out2_m + offs_n[None, :] * stride_out2_n
    out2_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N_half)
    tl.store(out2_ptrs, acc2.to(DTYPE_OUT), mask=out2_mask)


def _get_triton_dtype(torch_dtype):
    if torch_dtype == torch.float32:
        return tl.float32
    elif torch_dtype == torch.float16:
        return tl.float16
    elif torch_dtype == torch.bfloat16:
        return tl.bfloat16
    else:
        return tl.float32


@torch.fx.wrap
def fused_reshape_linear_slice(x_flat, weight_first, bias_first, weight_second, bias_second):
    M = x_flat.shape[0]
    K = x_flat.shape[1]
    N_half = weight_first.shape[0]

    # Both outputs are [M, 1, N_half] matching the 3D sliced output
    out_first = torch.empty(M, 1, N_half, dtype=x_flat.dtype, device=x_flat.device)
    out_second = torch.empty(M, 1, N_half, dtype=x_flat.dtype, device=x_flat.device)

    DTYPE_OUT = _get_triton_dtype(x_flat.dtype)

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32

    num_pid_m = (M + BLOCK_M - 1) // BLOCK_M
    num_pid_n = (N_half + BLOCK_N - 1) // BLOCK_N
    grid = (num_pid_m * num_pid_n,)

    fused_linear_slice_both_3d_kernel[grid](
        x_ptr=x_flat, weight1_ptr=weight_first, bias1_ptr=bias_first,
        weight2_ptr=weight_second, bias2_ptr=bias_second,
        out_first_ptr=out_first, out_second_ptr=out_second,
        M=M, K=K, N_half=N_half,
        stride_xm=x_flat.stride(0), stride_xk=x_flat.stride(1),
        stride_w1n=weight_first.stride(0), stride_w1k=weight_first.stride(1),
        stride_b1=bias_first.stride(0),
        stride_w2n=weight_second.stride(0), stride_w2k=weight_second.stride(1),
        stride_b2=bias_second.stride(0),
        stride_out1_m=out_first.stride(0), stride_out1_n=out_first.stride(2),
        stride_out2_m=out_second.stride(0), stride_out2_n=out_second.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        DTYPE_OUT=DTYPE_OUT,
    )

    # Return order matches pattern: tmp_11 (first half), tmp_12 (second half)
    return out_first, out_second


def replacement_func():
    return fused_reshape_linear_slice