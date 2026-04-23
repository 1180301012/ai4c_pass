import torch
import triton
import triton.language as tl


def pattern(in_2, in_1, in_0, in_3):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    transpose = linear.transpose(-1, -2)
    result = in_3 * transpose
    return result

def replacement_args(in_2, in_1, in_0, in_3):
    return (in_2, in_1, in_0, in_3)


@triton.jit
def fused_linear_transpose_mul_kernel(
    in2_ptr, in1_ptr, in0_ptr, in3_ptr, out_ptr,
    B, N, M, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    
    # Calculate offsets for M and N
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Process K dimension in chunks
    for k_start in range(0, K, 64):
        k_end = min(k_start + 64, K)
        k_mask = tl.arange(0, 64) < (k_end - k_start)
        
        # Load input (in_2: [B, N, K])
        in2 = tl.load(
            in2_ptr + pid_b * N * K + 
                   offs_n * K + k_start,
            mask=(mask_n[:, None] & k_mask[None, :]),
            other=0.0,
        )
        
        # Load weight (in_1: [M, K])
        in1 = tl.load(
            in1_ptr + offs_m[:, None] * K + k_start,
            mask=(mask_m[:, None] & k_mask[None, :]),
            other=0.0,
        )
        
        # Accumulate dot product
        acc += tl.dot(in1, in2, allow_tf32=True)

    # Add bias
    bias = tl.load(
        in0_ptr + offs_m,
        mask=mask_m,
        other=0.0,
    )
    bias = bias[:, None]
    acc += bias
    
    # Multiply by in_3
    in3 = tl.load(
        in3_ptr + pid_b * M * N + 
               offs_m[:, None] * N + offs_n,
        mask=(mask_m[:, None] & mask_n[None, :]),
        other=0.0,
    )
    acc *= in3

    # Store result
    tl.store(
        out_ptr + pid_b * M * N + 
               offs_m[:, None] * N + offs_n,
        acc,
        mask=(mask_m[:, None] & mask_n[None, :]),
    )


@torch.fx.wrap
def fused_linear_transpose_mul(in_2, in_1, in_0, in_3):
    B = in_2.size(0)
    N = in_2.size(1)
    K = in_2.size(2)
    M = in_1.size(0)
    
    out = torch.empty((B, M, N), dtype=in_2.dtype, device=in_2.device)
    
    BLOCK_M = 64
    BLOCK_N = 64
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    
    fused_linear_transpose_mul_kernel[
        (B, grid_m, grid_n)
    ](
        in_2, in_1, in_0, in_3, out,
        B, N, M, K,
        BLOCK_M, BLOCK_N,
    )
    
    return out

def replacement_func():
    return fused_linear_transpose_mul