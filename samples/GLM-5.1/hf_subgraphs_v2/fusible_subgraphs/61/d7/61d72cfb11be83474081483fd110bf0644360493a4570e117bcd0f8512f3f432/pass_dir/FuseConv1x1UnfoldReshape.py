import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    return (conv2d,)


def replacement_args(in_0, in_1):
    return (in_1, in_0)


@triton.jit
def pointwise_conv1x1_kernel(
    input_ptr, weight_ptr, output_ptr,
    M, N, K,
    stride_im, stride_ik,
    stride_on, stride_om,
    stride_wk, stride_wn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = off_m < M
    mask_n = off_n < N

    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        off_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = off_k < K

        w_ptrs = weight_ptr + off_n[:, None] * stride_wn + off_k[None, :] * stride_wk
        w = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)

        i_ptrs = input_ptr + off_k[:, None] * stride_ik + off_m[None, :] * stride_im
        i = tl.load(i_ptrs, mask=mask_k[:, None] & mask_m[None, :], other=0.0)

        acc += tl.dot(w, i, allow_tf32=True)

    o_ptrs = output_ptr + off_n[:, None] * stride_on + off_m[None, :] * stride_om
    tl.store(o_ptrs, acc, mask=mask_n[:, None] & mask_m[None, :])


@torch.fx.wrap
def pointwise_conv1x1(input, weight):
    C_out = weight.shape[0]
    C_in = weight.shape[1]
    H = input.shape[2]
    W = input.shape[3]
    M = H * W
    N = C_out
    K = C_in

    output = torch.empty(1, C_out, H, W, dtype=input.dtype, device=input.device)

    # Fixed block sizes for M=1024, N=128, K=256
    BLOCK_M = 64
    BLOCK_N = 32
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    pointwise_conv1x1_kernel[grid](
        input, weight, output,
        M, N, K,
        input.stride()[3],
        input.stride()[1],
        output.stride()[1],
        output.stride()[3],
        weight.stride()[1],
        weight.stride()[0],
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return output


def replacement_func():
    return pointwise_conv1x1