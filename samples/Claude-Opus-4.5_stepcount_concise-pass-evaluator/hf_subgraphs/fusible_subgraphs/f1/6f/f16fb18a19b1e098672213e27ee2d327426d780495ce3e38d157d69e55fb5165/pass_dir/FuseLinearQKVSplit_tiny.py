import torch
import triton
import triton.language as tl

# Pattern for linear operation
def pattern(weight, x):
    return torch.nn.functional.linear(x, weight, None)

def replacement_args(weight, x):
    return (weight, x)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
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
    # Simple 2D grid without swizzling
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    
    # Pointers
    x_ptrs = x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk
    w_ptrs = weight_ptr + rn[None, :] * stride_wn + rk[:, None] * stride_wk
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        # Load with boundary checks
        x_mask = (rm[:, None] < M) & ((k + rk[None, :]) < K)
        w_mask = (rn[None, :] < N) & ((k + rk[:, None]) < K)
        
        x_block = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_block = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc += tl.dot(x_block, w_block)
        
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk
    
    # Store result
    out_ptrs = out_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on
    out_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


@torch.fx.wrap
def triton_linear(weight, x):
    # Get dimensions - x is [1, 197, K], weight is [N, K]
    batch_size = x.shape[0]
    M = x.shape[1]  # seq_len
    K = x.shape[2]  # input_dim
    N = weight.shape[0]  # output_dim
    
    # Flatten x for GEMM: [1, 197, K] -> [197, K]
    x_2d = x.view(M, K).contiguous()
    weight_contig = weight.contiguous()
    
    # Allocate output
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    # Grid function for autotuning
    def grid(META):
        return (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
    
    matmul_kernel[grid](
        x_2d, weight_contig, out,
        M, N, K,
        x_2d.stride(0), x_2d.stride(1),
        weight_contig.stride(0), weight_contig.stride(1),
        out.stride(0), out.stride(1),
    )
    
    # Reshape back to [1, M, N]
    return out.view(batch_size, M, N)


def replacement_func():
    return triton_linear