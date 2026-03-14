import torch
import triton
import triton.language as tl

# Pattern for RMSNorm: cast to fp32 -> pow -> mean -> add eps -> rsqrt -> mul -> cast -> mul weight
def pattern(weight, x):
    tmp_10 = x.to(torch.float32)
    tmp_11 = tmp_10.pow(2)
    tmp_12 = tmp_11.mean(-1, keepdim=True)
    tmp_13 = tmp_12 + 1e-06
    tmp_14 = torch.rsqrt(tmp_13)
    tmp_15 = tmp_10 * tmp_14
    tmp_16 = tmp_15.to(torch.bfloat16)
    tmp_17 = weight * tmp_16
    return tmp_17

def replacement_args(weight, x):
    return (weight, x)

# RMSNorm kernel with autotuning - best 3 configs for correctness and performance
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=1),
    ],
    key=['N'],
)
@triton.jit
def fused_rmsnorm_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    stride_x,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * stride_x
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    mask = col_offsets < N
    x = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Compute sum of squares
    x_sq = x * x
    sum_sq = tl.sum(x_sq, axis=0)
    
    # Compute mean and rsqrt  
    mean_sq = sum_sq / N
    rstd = tl.rsqrt(mean_sq + eps)
    
    # Apply normalization and weight
    w = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    normalized = x * rstd
    out = w * normalized
    
    tl.store(out_ptr + row_start + col_offsets, out.to(tl.bfloat16), mask=mask)

@torch.fx.wrap
def fused_rmsnorm(weight, x):
    # RMSNorm computation
    orig_shape = x.shape
    hidden_dim = orig_shape[-1]
    x_2d = x.view(-1, hidden_dim)
    M = x_2d.shape[0]
    N = hidden_dim
    stride_x = x_2d.stride(0)
    
    rmsnorm_out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    eps = 1e-06
    
    grid = (M,)
    
    fused_rmsnorm_kernel[grid](
        x_2d,
        weight,
        rmsnorm_out,
        stride_x,
        N,
        eps,
    )
    
    return rmsnorm_out.view(orig_shape)

def replacement_func():
    return fused_rmsnorm