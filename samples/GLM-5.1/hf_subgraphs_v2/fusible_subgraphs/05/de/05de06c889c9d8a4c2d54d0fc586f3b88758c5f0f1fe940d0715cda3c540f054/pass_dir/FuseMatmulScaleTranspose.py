import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    matmul = torch.matmul(in_2, in_1)
    tmp_1 = matmul * in_0
    return tmp_1


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def fused_matmul_scale_kernel(
    in_0_ptr,
    in_1_ptr,
    in_2_ptr,
    out_ptr,
    M, K, N,
    stride_in2_m, stride_in2_k,
    stride_in1_k, stride_in1_n,
    stride_out_m, stride_out_n,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Load scalar
    scalar = tl.load(in_0_ptr)

    # Accumulate dot product in fp32
    acc = 0.0
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        a = tl.load(
            in_2_ptr + pid_m * stride_in2_m + k_offsets * stride_in2_k,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        b = tl.load(
            in_1_ptr + k_offsets * stride_in1_k + pid_n * stride_in1_n,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)

        acc += tl.sum(a * b)

    result = (acc * scalar.to(tl.float32)).to(out_ptr.dtype.element_ty)

    # Store to out[m, n]
    tl.store(out_ptr + pid_m * stride_out_m + pid_n * stride_out_n, result)


@torch.fx.wrap
def kernel_wrapper(in_0, in_1, in_2):
    M = in_2.shape[0]
    K = in_2.shape[1]
    N = in_1.shape[1]

    out = torch.empty((M, N), dtype=in_2.dtype, device=in_2.device)

    # Use a single BLOCK_K that covers K=512 in one iteration
    BLOCK_K = 512

    grid = (M, N)

    fused_matmul_scale_kernel[grid](
        in_0,
        in_1,
        in_2,
        out,
        M, K, N,
        in_2.stride(0), in_2.stride(1),
        in_1.stride(0), in_1.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_K=BLOCK_K,
    )

    return out


def replacement_func():
    return kernel_wrapper