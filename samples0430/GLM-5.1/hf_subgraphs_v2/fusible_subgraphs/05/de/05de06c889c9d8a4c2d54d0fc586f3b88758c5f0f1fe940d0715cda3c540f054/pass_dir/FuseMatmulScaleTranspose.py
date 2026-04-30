import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    matmul = torch.matmul(in_2, in_1)
    tmp_1 = matmul * in_0
    return (tmp_1,)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def fused_matmul_scale_kernel(
    in_2_ptr, in_1_ptr, in_0_ptr,
    out_ptr,
    M, N, K,
    stride_in2_m, stride_in2_k,
    stride_in1_k, stride_in1_n,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    acc = 0.0
    for k_start in range(0, K, BLOCK_K):
        offsets_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offsets_k < K

        a = tl.load(in_2_ptr + pid_m * stride_in2_m + offsets_k * stride_in2_k, mask=mask_k, other=0.0)
        b = tl.load(in_1_ptr + offsets_k * stride_in1_k + pid_n * stride_in1_n, mask=mask_k, other=0.0)

        acc = acc + tl.sum(a * b)

    scale = tl.load(in_0_ptr)
    result = acc * scale

    tl.store(out_ptr + pid_m * N + pid_n, result)


@torch.fx.wrap
def fused_matmul_scale(in_0, in_1, in_2):
    M, K = in_2.shape
    K2, N = in_1.shape

    dtype = in_2.dtype
    device = in_2.device

    out = torch.empty((M, N), dtype=dtype, device=device)

    BLOCK_K = 512

    grid = (M, N)

    fused_matmul_scale_kernel[grid](
        in_2_ptr=in_2, in_1_ptr=in_1, in_0_ptr=in_0,
        out_ptr=out,
        M=M, N=N, K=K,
        stride_in2_m=in_2.stride(0), stride_in2_k=in_2.stride(1),
        stride_in1_k=in_1.stride(0), stride_in1_n=in_1.stride(1),
        BLOCK_K=BLOCK_K,
    )

    return out


def replacement_func():
    return fused_matmul_scale