import torch
import triton
import triton.language as tl

@triton.jit
def scale_and_softmax_kernel(
    in_ptr,
    out_ptr,
    B,
    M,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    
    block_start_m = pid_m * BLOCK_SIZE_M
    
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, K)
    
    input = tl.load(
        in_ptr + pid_b * M * K + offs_m[:, None] * K + offs_k[None, :],
        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
        other=0.0,
    )
    
    input_scaled = input * 0.0625
    
    max_val = tl.max(input_scaled, axis=1)
    input_sub = input_scaled - max_val[:, None]
    exp_input = tl.exp(input_sub)
    sum_exp = tl.sum(exp_input, axis=1)
    output = exp_input / sum_exp[:, None]
    
    tl.store(
        out_ptr + pid_b * M * K + offs_m[:, None] * K + offs_k[None, :],
        output,
        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
    )

@torch.fx.wrap
def scale_and_softmax(in_0):
    B, M, K = in_0.shape
    BLOCK_SIZE_M = 256
    num_programs_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    out = torch.empty_like(in_0)
    scale_and_softmax_kernel[(B, num_programs_m)](
        in_0, out, B, M, K, BLOCK_SIZE_M, 32
    )
    return out

def pattern(in_0):
    tmp0 = 0.0625 * in_0
    tmp1 = torch.nn.functional.softmax(tmp0, dim=-1)
    return tmp1

def replacement_args(in_0):
    return (in_0,)

def replacement_func():
    return scale_and_softmax