import torch
import triton
import triton.language as tl

# Pattern for second linear: reshape(in_4) -> linear -> slice both halves
def pattern(bias, weight, x):
    reshaped = x.reshape(300, -1, 256)
    linear_out = torch.nn.functional.linear(reshaped, weight, bias)
    first_half = linear_out[Ellipsis, slice(None, 256, None)]
    second_half = linear_out[Ellipsis, slice(-256, None, None)]
    return first_half, second_half

def replacement_args(bias, weight, x):
    return (bias, weight, x)


@triton.jit
def linear_full_kernel(
    X_ptr,      # [M, K] input
    W_ptr,      # [N, K] weight (N = 512)
    B_ptr,      # [N] bias
    Out_ptr,    # [M, N] output
    M, K, N,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    # Single iteration since BLOCK_K = K = 256
    offs_k = tl.arange(0, BLOCK_K)
    
    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    x = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0)
    
    w_ptrs = W_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
    w = tl.load(w_ptrs, mask=mask_n[None, :], other=0.0)
    
    acc = tl.dot(x, w)
    
    b = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + b[None, :]
    
    mask = mask_m[:, None] & mask_n[None, :]
    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    
    tl.store(out_ptrs, acc, mask=mask)


@torch.fx.wrap
def compute_full_linear(bias, weight, x):
    reshaped = x.reshape(300, -1, 256)
    x_2d = reshaped.view(300, 256)
    
    M, K, N = 300, 256, 512
    out = torch.empty((M, 1, N), device=x.device, dtype=x.dtype)
    out_2d = out.view(M, N)
    
    # Optimized for M=300, K=256, N=512
    BLOCK_M, BLOCK_K, BLOCK_N = 32, 256, 64
    grid = ((M + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N)
    
    linear_full_kernel[grid](
        x_2d, weight, bias, out_2d,
        M, K, N,
        x_2d.stride(0), x_2d.stride(1),
        weight.stride(0), weight.stride(1),
        out_2d.stride(0), out_2d.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
    )
    
    return out


def replacement_impl(bias, weight, x):
    full = compute_full_linear(bias, weight, x)
    first = full[Ellipsis, slice(None, 256, None)]
    second = full[Ellipsis, slice(-256, None, None)]
    return first, second


def replacement_func():
    return replacement_impl