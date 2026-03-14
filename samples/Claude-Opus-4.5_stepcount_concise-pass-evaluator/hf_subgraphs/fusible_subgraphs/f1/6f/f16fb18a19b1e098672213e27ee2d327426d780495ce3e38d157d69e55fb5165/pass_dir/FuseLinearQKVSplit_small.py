import torch
import triton
import triton.language as tl

# Pattern for linear operation
def pattern(weight, x):
    return torch.nn.functional.linear(x, weight, None)

def replacement_args(weight, x):
    return (weight, x)


@triton.jit
def matmul_kernel(
    x_ptr, weight_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        k_offs = k + rk
        
        x_ptrs = x_ptr + rm[:, None] * stride_xm + k_offs[None, :] * stride_xk
        x_mask = (rm[:, None] < M) & (k_offs[None, :] < K)
        x_block = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        w_ptrs = weight_ptr + rn[None, :] * stride_wn + k_offs[:, None] * stride_wk
        w_mask = (rn[None, :] < N) & (k_offs[:, None] < K)
        w_block = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc += tl.dot(x_block, w_block)
    
    out_ptrs = out_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on
    out_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


@torch.fx.wrap
def triton_linear(weight, x):
    # Get dimensions - x is [1, seq_len, K], weight is [N, K]
    batch_size = x.shape[0]
    M = x.shape[1]  # seq_len
    K = x.shape[2]  # input_dim
    N = weight.shape[0]  # output_dim
    
    # Flatten x for GEMM: [1, seq_len, K] -> [seq_len, K]
    x_2d = x.view(M, K)
    
    # Allocate output
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    # Launch GEMM kernel with fixed block sizes
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    
    matmul_kernel[(grid_m, grid_n)](
        x_2d, weight, out,
        M, N, K,
        x_2d.stride(0), x_2d.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
    )
    
    # Reshape back to [batch_size, M, N]
    return out.view(batch_size, M, N)


def replacement_func():
    return triton_linear