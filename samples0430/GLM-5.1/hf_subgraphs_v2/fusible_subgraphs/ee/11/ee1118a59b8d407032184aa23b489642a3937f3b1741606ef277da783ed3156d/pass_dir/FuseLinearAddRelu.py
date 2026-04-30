import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = in_2 + linear
    tmp_4 = tmp_3.relu_()
    return (tmp_4,)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_linear_add_relu_kernel(
    input_ptr, weight_ptr, bias_ptr, residual_ptr, output_ptr,
    M, N, K,
    stride_im, stride_ik,
    stride_wn, stride_wk,
    stride_bn,
    stride_rm, stride_rn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    # Accumulator for matmul result (float32 for precision)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension for matrix multiplication
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Load input tile: [BLOCK_M, BLOCK_K]
        a_ptrs = input_ptr + offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik
        a_mask = mask_m[:, None] & mask_k[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load weight tile (transposed for linear op): [BLOCK_K, BLOCK_N]
        # torch.nn.functional.linear computes x @ W.T + b
        # We load W.T directly by arranging offsets as [BLOCK_K, BLOCK_N]
        b_ptrs = weight_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        b_mask = mask_k[:, None] & mask_n[None, :]
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b, allow_tf32=True)

    # Load bias: [BLOCK_N] - broadcast over M dimension
    bias_ptrs = bias_ptr + offs_n * stride_bn
    bias = tl.load(bias_ptrs, mask=mask_n, other=0.0).to(tl.float32)
    acc += bias[None, :]

    # Load residual: [BLOCK_M, BLOCK_N]
    r_ptrs = residual_ptr + offs_m[:, None] * stride_rm + offs_n[None, :] * stride_rn
    r_mask = mask_m[:, None] & mask_n[None, :]
    residual = tl.load(r_ptrs, mask=r_mask, other=0.0).to(tl.float32)
    acc += residual

    # ReLU activation
    acc = tl.where(acc > 0, acc, 0.0)

    # Store output
    o_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    o_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(o_ptrs, acc, mask=o_mask)


@torch.fx.wrap
def fused_linear_add_relu(bias, weight, residual, input_tensor):
    # Ensure weight and bias are on the same device as input
    device = input_tensor.device
    if weight.device != device:
        weight = weight.to(device)
    if bias.device != device:
        bias = bias.to(device)

    M = input_tensor.shape[0]
    K = input_tensor.shape[1]
    N = weight.shape[0]  # weight shape is [N, K] = [out_features, in_features]

    # Allocate output tensor
    output = torch.empty((M, N), dtype=input_tensor.dtype, device=device)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )

    fused_linear_add_relu_kernel[grid](
        input_ptr=input_tensor, weight_ptr=weight, bias_ptr=bias,
        residual_ptr=residual, output_ptr=output,
        M=M, N=N, K=K,
        stride_im=input_tensor.stride(0), stride_ik=input_tensor.stride(1),
        stride_wn=weight.stride(0), stride_wk=weight.stride(1),
        stride_bn=bias.stride(0),
        stride_rm=residual.stride(0), stride_rn=residual.stride(1),
        stride_om=output.stride(0), stride_on=output.stride(1),
    )

    return (output,)


def replacement_func():
    return fused_linear_add_relu