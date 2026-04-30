import torch
import triton
import triton.language as tl


# Fixed Triton kernel for mean reduction - optimized for N=448
@triton.jit
def mean_dim2_kernel(
    mean_input_ptr, mean_output_ptr,
    M, K, N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Mean reduction along dim=-2: input [M, K, N] -> output [M, N]
    BLOCK_SIZE=256 is optimal for N=448
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_n = offs_n < N
    
    # Accumulate along K dimension
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for k in range(K):
        ptr = mean_input_ptr + pid_m * K * N + k * N + offs_n
        tile = tl.load(ptr, mask=mask_n, other=0.0)
        acc = acc + tile
    
    mean_val = acc / K
    
    # Store output
    out_ptr = mean_output_ptr + pid_m * N + offs_n
    tl.store(out_ptr, mean_val, mask=mask_n)


@torch.fx.wrap
def triton_mean_reduction(in_3):
    """
    Optimized mean operation along dim=-2 using Triton.
    """
    M, K, N = in_3.shape
    mean_out = torch.empty((M, N), dtype=in_3.dtype, device=in_3.device)
    
    # Fixed BLOCK_SIZE=256 for optimal memory coalescing with N=448
    BLOCK_SIZE = 256
    grid = (M, (N + BLOCK_SIZE - 1) // BLOCK_SIZE, 1)
    
    mean_dim2_kernel[grid](
        in_3, mean_out,
        M, K, N,
        BLOCK_SIZE,
    )
    
    return mean_out


def pattern(in_3):
    """
    Match the mean reduction pattern.
    """
    return in_3.mean(-2)


def replacement_args(in_3):
    return (in_3,)


def replacement_func():
    return triton_mean_reduction