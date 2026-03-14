import torch
import triton
import triton.language as tl

# Pattern matching function - must match model.py exactly
def pattern(bias, weight, x):
    conv_out = torch.conv2d(x, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    hardswish_out = torch.nn.functional.hardswish(conv_out, True)
    flatten_out = hardswish_out.flatten(1, -1)
    return flatten_out

# Argument extraction function
def replacement_args(bias, weight, x):
    return (bias, weight, x)

# Triton kernel for batch size 1 - smaller blocks for better parallelism
@triton.jit
def fused_gemv_hardswish_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    N, K,
    stride_wn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N
    
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_base = weight_ptr + n_offs[:, None] * stride_wn
    
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K
        
        x_tile = tl.load(x_ptr + k_offs, mask=k_mask, other=0.0)
        w_tile = tl.load(w_base + k_offs[None, :], mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        
        acc += tl.sum(w_tile * x_tile[None, :], axis=1)
    
    bias_val = tl.load(bias_ptr + n_offs, mask=n_mask, other=0.0)
    acc = acc + bias_val
    
    # hardswish
    x_plus_3 = acc + 3.0
    relu6 = tl.minimum(tl.maximum(x_plus_3, 0.0), 6.0)
    out = acc * relu6 * 0.16666666666666666
    
    tl.store(out_ptr + n_offs, out, mask=n_mask)

# Simple 2D grid kernel for matmul
@triton.jit
def fused_matmul_hardswish_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_wn, stride_om,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    m_mask = m_offs < M
    n_mask = n_offs < N
    
    x_base = x_ptr + m_offs[:, None] * stride_xm
    w_base = weight_ptr + n_offs[:, None] * stride_wn
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K
        
        x_tile = tl.load(x_base + k_offs[None, :], mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        w_tile = tl.load(w_base + k_offs[None, :], mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        acc = tl.dot(x_tile, tl.trans(w_tile), acc, input_precision="tf32")
    
    bias_val = tl.load(bias_ptr + n_offs, mask=n_mask, other=0.0)
    acc = acc + bias_val[None, :]
    
    # hardswish
    x_plus_3 = acc + 3.0
    relu6 = tl.minimum(tl.maximum(x_plus_3, 0.0), 6.0)
    out = acc * relu6 * 0.16666666666666666
    
    out_ptrs = out_ptr + m_offs[:, None] * stride_om + n_offs[None, :]
    tl.store(out_ptrs, out, mask=m_mask[:, None] & n_mask[None, :])

@torch.fx.wrap
def fused_conv1x1_hardswish_flatten(bias, weight, x):
    B = x.shape[0]
    C_in = x.shape[1]
    C_out = weight.shape[0]
    
    x_2d = x.view(B, C_in)
    weight_2d = weight.view(C_out, C_in)
    out = torch.empty((B, C_out), device=x.device, dtype=x.dtype)
    
    M, K, N = B, C_in, C_out
    
    if M == 1:
        # Smaller blocks for more parallelism
        BLOCK_N, BLOCK_K = 64, 32
        grid = (triton.cdiv(N, BLOCK_N),)
        fused_gemv_hardswish_kernel[grid](
            x_2d, weight_2d, bias, out,
            N, K,
            weight_2d.stride(0),
            BLOCK_N, BLOCK_K,
        )
    else:
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        fused_matmul_hardswish_kernel[grid](
            x_2d, weight_2d, bias, out,
            M, N, K,
            x_2d.stride(0), weight_2d.stride(0), out.stride(0),
            BLOCK_M, BLOCK_N, BLOCK_K,
        )
    
    return out

def replacement_func():
    return fused_conv1x1_hardswish_flatten