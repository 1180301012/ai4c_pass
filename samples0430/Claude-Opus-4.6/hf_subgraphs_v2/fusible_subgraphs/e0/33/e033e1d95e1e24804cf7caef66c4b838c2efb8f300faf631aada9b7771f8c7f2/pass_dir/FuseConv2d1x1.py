import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    return conv2d


def replacement_args(in_0, in_1, in_3):
    return (in_3, in_1, in_0)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=5),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=5),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_conv1x1_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    B, M, N, K,
    stride_input_b, stride_input_k, stride_input_n,
    stride_weight_m, stride_weight_k,
    stride_output_b, stride_output_m, stride_output_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_b = tl.program_id(1)

    num_m_blocks = tl.cdiv(M, BLOCK_M)
    num_n_blocks = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_SIZE_M * num_n_blocks
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_m_blocks - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    input_base = pid_b * stride_input_b

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        # Load weight tile [BLOCK_M, BLOCK_K]
        w_ptrs = weight_ptr + m_offsets[:, None] * stride_weight_m + k_offsets[None, :] * stride_weight_k
        w_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Load input tile [BLOCK_K, BLOCK_N]
        i_ptrs = input_ptr + input_base + k_offsets[:, None] * stride_input_k + n_offsets[None, :] * stride_input_n
        i_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        x = tl.load(i_ptrs, mask=i_mask, other=0.0)

        acc += tl.dot(w, x)

    # Add bias
    bias = tl.load(bias_ptr + m_offsets, mask=m_offsets < M, other=0.0)
    acc += bias[:, None]

    # Store output
    output_base = pid_b * stride_output_b
    o_ptrs = output_ptr + output_base + m_offsets[:, None] * stride_output_m + n_offsets[None, :] * stride_output_n
    o_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(o_ptrs, acc.to(output_ptr.dtype.element_ty), mask=o_mask)


@torch.fx.wrap
def conv1x1_gemm(input_tensor, weight, bias):
    B = input_tensor.shape[0]
    C_in = input_tensor.shape[1]
    H = input_tensor.shape[2]
    W = input_tensor.shape[3]
    C_out = weight.shape[0]
    N = H * W
    M = C_out
    K = C_in

    output = torch.empty((B, C_out, H, W), dtype=input_tensor.dtype, device=input_tensor.device)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']), B)

    gemm_conv1x1_kernel[grid](
        input_tensor, weight, bias, output,
        B, M, N, K,
        input_tensor.stride(0), input_tensor.stride(1), 1,
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1), 1,
    )

    return output


def replacement_func():
    return conv1x1_gemm