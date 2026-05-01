import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_2, in_1, in_0):
    tmp_2 = torch.nn.functional.dropout(in_2, p=0.0, training=False)
    to = tmp_2.to(torch.float16)
    linear = torch.nn.functional.linear(to, in_1, in_0)
    return linear

# Argument extraction function
def replacement_args(in_2, in_1, in_0):
    return (to, in_1, in_0)

# Triton kernel for linear operation
@triton.jit
def linear_kernel(A, B, C, bias, m, n, k,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    block_start_m = pid_m * BLOCK_M
    block_start_n = pid_n * BLOCK_N

    offs_m = block_start_m + tl.arange(0, BLOCK_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = (offs_m < m)[:, None]
    mask_n = (offs_n < n)[None, :]
    mask = mask_m & mask_n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, k, BLOCK_K):
        a = tl.load(
            A + (block_start_m + offs_m)[:, None] * stride_A_m + (k_start + offs_k) * stride_A_k,
            mask=mask_m & (k_start + offs_k < k),
            other=0.0
        )
        b = tl.load(
            B + (k_start + offs_k)[:, None] * stride_B_k + (block_start_n + offs_n) * stride_B_n,
            mask=(k_start + offs_k < k)[:, None] & mask_n,
            other=0.0
        )
        acc += tl.dot(a, b)

    out = acc.to(tl.float16)
    out += bias
    tl.store(
        C + (block_start_m + offs_m)[:, None] * stride_C_m + (block_start_n + offs_n) * stride_C_n,
        out,
        mask=mask
    )

# Kernel wrapper for Triton
@torch.fx.wrap
def linear_wrapper(to, in_1, in_0):
    batch, seq_len, in_features = to.shape
    m = batch * seq_len
    n = in_1.shape[0]
    k = in_features

    input_2d = to.reshape(m, k)
    weight_T = in_1.t()
    out_2d = torch.empty((m, n), dtype=to.dtype)

    # Set kernel configuration
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64

    num_blocks_m = (m + BLOCK_M - 1) // BLOCK_M
    num_blocks_n = (n + BLOCK_N - 1) // BLOCK_N

    # Get tensor strides (assuming contiguous)
    stride_A_m = input_2d.stride(0)
    stride_A_k = input_2d.stride(1)
    stride_B_k = weight_T.stride(0)
    stride_B_n = weight_T.stride(1)
    stride_C_m = out_2d.stride(0)
    stride_C_n = out_2d.stride(1)

    # Launch kernel
    linear_kernel[(num_blocks_m, num_blocks_n)](
        input_2d,
        weight_T,
        out_2d,
        in_0,
        m, n, k,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K
    )

    return out_2d.view(batch, seq_len, n)

# Replacement function
def replacement_func():
    return linear_wrapper