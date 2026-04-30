import torch
import triton
import triton.language as tl


def pattern(bias, weight, input_tensor):
    conv2d = torch.conv2d(input_tensor, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.silu(conv2d, inplace=False)
    tmp_4 = torch.nn.functional.dropout(tmp_3, 0.0, False, False)
    return tmp_4


def replacement_args(bias, weight, input_tensor):
    return (bias, weight, input_tensor)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_stages=2, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_conv1x1_silu_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    M, N, K,
    HW,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # M = batch * H * W (spatial), N = C_out, K = C_in
    # Input layout: [batch, C_in, H, W] contiguous NCHW
    #   A[m, k] = input_ptr[m + k * HW]  (stride_m=1, stride_k=HW)
    # Weight layout: [C_out, C_in, 1, 1] contiguous
    #   B[n, k] = weight_ptr[n * K + k]
    # Output layout: [batch, C_out, H, W] contiguous NCHW
    #   C[m, n] = output_ptr[m + n * HW]  (stride_m=1, stride_n=HW)
    # Compute: C = A @ B^T + bias, then SiLU

    pid = tl.program_id(0)
    num_n_blocks = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_n_blocks
    pid_n = pid % num_n_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load A tile [BLOCK_M, BLOCK_K]
        a_ptrs = input_ptr + offs_m[:, None] + offs_k[None, :] * HW
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B^T tile [BLOCK_K, BLOCK_N] where B^T[k, n] = weight[n, k]
        b_ptrs = weight_ptr + offs_k[:, None] + offs_n[None, :] * K
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

    # Add bias
    bias_vals = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc += bias_vals[None, :]

    # SiLU: x * sigmoid(x)
    result = acc * tl.sigmoid(acc)

    # Store result
    c_ptrs = output_ptr + offs_m[:, None] + offs_n[None, :] * HW
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, result, mask=c_mask)


@torch.fx.wrap
def fused_conv1x1_silu(bias, weight, input_tensor):
    batch = input_tensor.shape[0]
    C_in = input_tensor.shape[1]
    H = input_tensor.shape[2]
    W = input_tensor.shape[3]
    C_out = weight.shape[0]

    M = batch * H * W
    N = C_out
    K = C_in
    HW = H * W

    output = torch.empty((batch, C_out, H, W), dtype=input_tensor.dtype, device=input_tensor.device)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

    fused_conv1x1_silu_kernel[grid](
        input_tensor, weight, bias, output,
        M, N, K,
        HW,
    )

    return output


def replacement_func():
    return fused_conv1x1_silu