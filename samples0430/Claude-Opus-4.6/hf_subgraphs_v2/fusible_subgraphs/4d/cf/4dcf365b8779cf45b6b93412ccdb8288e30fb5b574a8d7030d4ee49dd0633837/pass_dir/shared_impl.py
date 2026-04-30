import torch
import triton
import triton.language as tl


@triton.jit
def matmul_bias_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    M, N, K,
    stride_im, stride_wn,
    stride_om,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    MASK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = input_ptr + offs_m[:, None] * stride_im + offs_k[None, :]
    b_ptrs = weight_ptr + offs_n[None, :] * stride_wn + offs_k[:, None]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        if MASK_M:
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M), other=0.0)
        else:
            a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K

    # Add bias
    bias = tl.load(bias_ptr + offs_n)
    acc += bias[None, :]

    # Store output
    c_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
    if MASK_M:
        tl.store(c_ptrs, acc.to(output_ptr.dtype.element_ty), mask=(offs_m[:, None] < M))
    else:
        tl.store(c_ptrs, acc.to(output_ptr.dtype.element_ty))


def _do_matmul_bias(in_0, in_1, in_2):
    input_shape = in_2.shape
    K = in_1.shape[1]
    N = in_1.shape[0]
    M = in_2.numel() // K
    out_shape = input_shape[:-1] + (N,)
    output = torch.empty(out_shape, dtype=in_1.dtype, device=in_2.device)

    # Choose block sizes based on problem dimensions
    if M <= 32:
        # Thin M (bigbird-like: M=17, K=768, N=3072)
        # Grid = (1, 48), K_loop=12 => 48 programs for good SM utilization
        BLOCK_M = 32
        BLOCK_N = 64
        BLOCK_K = 64
        MASK_M = True
        num_warps = 4
        num_stages = 2
    else:
        # Square-ish (RECT_L-like: M=128, K=128, N=128)
        # Grid = (4, 4)=16, K_loop=1 => quick single-pass
        BLOCK_M = 32
        BLOCK_N = 32
        BLOCK_K = 128
        MASK_M = (M % 32 != 0)
        num_warps = 4
        num_stages = 1

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_bias_kernel[grid](
        in_2, in_1, in_0, output,
        M, N, K,
        in_2.stride(-2), in_1.stride(0),
        output.stride(-2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        MASK_M=MASK_M,
        num_warps=num_warps, num_stages=num_stages,
    )
    return output


@torch.fx.wrap
def fused_linear_dispatch(in_0, in_1, in_2, route):
    return _do_matmul_bias(in_0, in_1, in_2)