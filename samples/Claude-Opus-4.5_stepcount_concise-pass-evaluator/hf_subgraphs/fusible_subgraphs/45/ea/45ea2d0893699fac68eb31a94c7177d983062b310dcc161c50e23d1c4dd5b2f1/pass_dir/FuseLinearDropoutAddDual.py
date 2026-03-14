import torch
import triton
import triton.language as tl

# Pattern for LINKX-style: linear -> dropout(p=0.0, training=False) -> residual + result
def pattern(in_0, in_1, in_2, in_3):
    linear_out = torch.nn.functional.linear(in_2, in_1, in_0)
    dropout_out = torch.nn.functional.dropout(linear_out, p=0.0, training=False)
    sum_out = in_3 + dropout_out
    return (sum_out, dropout_out)

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_linear_bias_residual_dual_kernel(
    input_ptr, weight_ptr, bias_ptr, residual_ptr,
    linear_out_ptr, sum_out_ptr,
    M, N, K, num_pid_n,
    stride_im, stride_ik,
    stride_wn, stride_wk,
    stride_rm, stride_rn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        
        a_mask = (rm[:, None] < M) & (rk[None, :] < K)
        a = tl.load(input_ptr + rm[:, None] * stride_im + rk[None, :] * stride_ik, 
                    mask=a_mask, other=0.0)
        
        b_mask = (rk[:, None] < K) & (rn[None, :] < N)
        b = tl.load(weight_ptr + rn[None, :] * stride_wn + rk[:, None] * stride_wk,
                    mask=b_mask, other=0.0)
        
        acc += tl.dot(a, b)
    
    bias_mask = rn < N
    bias_vals = tl.load(bias_ptr + rn, mask=bias_mask, other=0.0)
    acc = acc + bias_vals[None, :]
    
    res_mask = (rm[:, None] < M) & (rn[None, :] < N)
    residual_vals = tl.load(residual_ptr + rm[:, None] * stride_rm + rn[None, :] * stride_rn,
                            mask=res_mask, other=0.0)
    
    linear_out = acc
    sum_out = residual_vals + linear_out
    
    out_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(linear_out_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on,
             linear_out, mask=out_mask)
    tl.store(sum_out_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on,
             sum_out, mask=out_mask)

@torch.fx.wrap
def fused_linear_residual_add_dual(in_0, in_1, in_2, in_3):
    # in_0=bias, in_1=weight, in_2=input, in_3=residual
    M, K = in_2.shape
    N = in_1.shape[0]
    
    inp = in_2.contiguous()
    weight = in_1.contiguous()
    residual = in_3.contiguous()
    bias = in_0
    
    linear_out = torch.empty((M, N), device=inp.device, dtype=inp.dtype)
    sum_out = torch.empty((M, N), device=inp.device, dtype=inp.dtype)
    
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64
    
    num_pid_m = (M + BLOCK_M - 1) // BLOCK_M
    num_pid_n = (N + BLOCK_N - 1) // BLOCK_N
    grid = (num_pid_m * num_pid_n,)
    
    fused_linear_bias_residual_dual_kernel[grid](
        inp, weight, bias, residual,
        linear_out, sum_out,
        M, N, K, num_pid_n,
        inp.stride(0), inp.stride(1),
        weight.stride(0), weight.stride(1),
        residual.stride(0), residual.stride(1),
        linear_out.stride(0), linear_out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    
    return (sum_out, linear_out)

def replacement_func():
    return fused_linear_residual_add_dual