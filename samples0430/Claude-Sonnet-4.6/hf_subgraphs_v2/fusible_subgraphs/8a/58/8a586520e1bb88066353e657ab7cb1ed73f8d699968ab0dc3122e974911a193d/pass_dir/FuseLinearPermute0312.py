import torch
import triton
import triton.language as tl


def pattern(input, weight, bias):
    linear = torch.nn.functional.linear(input, weight, bias)
    out = linear.permute(0, 3, 1, 2)
    return out


def replacement_args(input, weight, bias):
    return (input, weight, bias)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64}),
        triton.Config({'BLOCK_M': 128}),
        triton.Config({'BLOCK_M': 256}),
        triton.Config({'BLOCK_M': 512}),
    ],
    key=['M'],
)
@triton.jit
def _fused_linear_permute_0312_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M,
    K: tl.constexpr,
    N: tl.constexpr,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    # Loop over each output feature n (fully unrolled since N is constexpr)
    for n in tl.static_range(N):
        dot = tl.zeros((BLOCK_M,), dtype=tl.float32)
        # Loop over input features k (fully unrolled since K is constexpr)
        for k in tl.static_range(K):
            # x[m, k] for m in offs_m: stride is K since last dim is K
            x_k = tl.load(x_ptr + offs_m * K + k, mask=mask_m, other=0.0).to(tl.float32)
            # w[n, k]: scalar broadcast
            w_nk = tl.load(w_ptr + n * K + k).to(tl.float32)
            dot = dot + x_k * w_nk
        # Add bias
        b_n = tl.load(b_ptr + n).to(tl.float32)
        dot = dot + b_n

        # Cast to output dtype
        if IS_FP16:
            out_val = dot.to(tl.float16)
        elif IS_BF16:
            out_val = dot.to(tl.bfloat16)
        else:
            out_val = dot

        # Store at output[n, m]: permuted layout [N, M]
        tl.store(out_ptr + n * M + offs_m, out_val, mask=mask_m)


@torch.fx.wrap
def fused_linear_permute_0_3_1_2(input, weight, bias):
    # input:  [B, I, J, K]  e.g. [1, 196, 196, 3]
    # weight: [N, K]        e.g. [16, 3]
    # bias:   [N]           e.g. [16]
    # output: [B, N, I, J]  e.g. [1, 16, 196, 196]
    B = input.shape[0]
    I = input.shape[1]
    J = input.shape[2]
    K = input.shape[3]
    N = weight.shape[0]
    M = B * I * J  # flattened spatial size

    # Move weight/bias to same device as input (they may be on CPU)
    weight_dev = weight.to(device=input.device)
    bias_dev = bias.to(device=input.device)

    out = torch.empty((B, N, I, J), dtype=input.dtype, device=input.device)

    IS_FP16 = input.dtype == torch.float16
    IS_BF16 = input.dtype == torch.bfloat16

    grid = lambda meta: ((M + meta['BLOCK_M'] - 1) // meta['BLOCK_M'],)

    _fused_linear_permute_0312_kernel[grid](
        input, weight_dev, bias_dev, out,
        M, K, N,
        IS_FP16=IS_FP16,
        IS_BF16=IS_BF16,
    )

    return out


def replacement_func():
    return fused_linear_permute_0_3_1_2