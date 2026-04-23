import torch
import triton
import triton.language as tl


# ---- Triton kernel for linear (matmul + bias) ----

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=2),
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=2),
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 4, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=2),
        triton.Config({'BLOCK_M': 4, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 4, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    M, N, K,
    stride_im, stride_ik,
    stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    # Load bias for this output block
    bias_ptrs = bias_ptr + offs_n
    bias_vals = tl.load(bias_ptrs, mask=mask_n, other=0.0).to(tl.float32)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32) + bias_vals

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k
        mask_k = k_offs < K

        # Load input: [BLOCK_M, BLOCK_K]
        input_ptrs = input_ptr + offs_m[:, None] * stride_im + k_offs[None, :] * stride_ik
        input_vals = tl.load(input_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)

        # Load weight: [BLOCK_K, BLOCK_N]
        weight_ptrs = weight_ptr + k_offs[:, None] * stride_wk + offs_n[None, :] * stride_wn
        weight_vals = tl.load(weight_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0).to(tl.float32)

        acc += tl.dot(input_vals, weight_vals)

    # Store output
    output_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(output_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


# ---- Triton kernel for batch_norm (inference mode) ----

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 64}, num_warps=2),
        triton.Config({'BLOCK_C': 128}, num_warps=2),
        triton.Config({'BLOCK_C': 256}, num_warps=4),
        triton.Config({'BLOCK_C': 384}, num_warps=4),
        triton.Config({'BLOCK_C': 512}, num_warps=8),
    ],
    key=['C'],
)
@triton.jit
def batch_norm_inference_kernel(
    input_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr, output_ptr,
    B, C,
    BLOCK_C: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= B:
        return

    c_offsets = tl.arange(0, BLOCK_C)
    mask = c_offsets < C

    # Load parameters
    running_mean = tl.load(running_mean_ptr + c_offsets, mask=mask, other=0.0).to(tl.float32)
    running_var = tl.load(running_var_ptr + c_offsets, mask=mask, other=1.0).to(tl.float32)
    weight = tl.load(weight_ptr + c_offsets, mask=mask, other=1.0).to(tl.float32)
    bias = tl.load(bias_ptr + c_offsets, mask=mask, other=0.0).to(tl.float32)

    eps = 1e-05
    inv_std = 1.0 / tl.sqrt(running_var + eps)
    scale = weight * inv_std
    shift = bias - running_mean * scale

    input_row = tl.load(input_ptr + row_idx * C + c_offsets, mask=mask, other=0.0).to(tl.float32)
    output_row = input_row * scale + shift
    tl.store(output_ptr + row_idx * C + c_offsets, output_row, mask=mask)


@torch.fx.wrap
def fused_dispatch_wrapper(*args):
    route = args[-1]
    args = args[:-1]

    if route == "linear":
        input_tensor, weight, bias = args
        M = input_tensor.shape[0]
        K = input_tensor.shape[1]
        N = weight.shape[0]

        output = torch.empty((M, N), dtype=input_tensor.dtype, device=input_tensor.device)

        grid = (triton.cdiv(M, 1), triton.cdiv(N, 32))
        linear_kernel[grid](
            input_tensor, weight, bias, output,
            M, N, K,
            input_tensor.stride(0), input_tensor.stride(1),
            weight.stride(1), weight.stride(0),
            output.stride(0), output.stride(1),
        )
        return output

    elif route == "bn":
        input_tensor, running_mean, running_var, bn_weight, bn_bias = args
        B = input_tensor.shape[0]
        C = input_tensor.shape[1]

        output = torch.empty((B, C), dtype=input_tensor.dtype, device=input_tensor.device)

        batch_norm_inference_kernel[(B,)](
            input_tensor, running_mean, running_var, bn_weight, bn_bias, output,
            B, C,
        )
        return output

    else:
        raise ValueError(f"Unknown route: {route}")


def pattern(in_6, in_5, in_4):
    linear = torch.nn.functional.linear(in_6, in_5, in_4)
    return linear


def replacement_args(in_6, in_5, in_4):
    return (in_6, in_5, in_4, "linear")


def replacement_func():
    return fused_dispatch_wrapper