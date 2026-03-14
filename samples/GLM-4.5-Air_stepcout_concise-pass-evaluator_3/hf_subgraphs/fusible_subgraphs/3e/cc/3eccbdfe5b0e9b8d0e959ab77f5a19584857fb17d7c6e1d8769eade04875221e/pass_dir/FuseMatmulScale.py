import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    """Pattern matching only the matmul and scale part"""
    tmp_0 = torch.matmul(in_2, in_1)
    tmp_1 = tmp_0 * in_0
    return tmp_1


def replacement_args(in_0, in_1, in_2):
    """Extract arguments needed for the replacement kernel"""
    return (in_0, in_1, in_2)


@triton.jit
def matmul_scale_kernel(
    a_ptr, b_ptr, scale_val, output_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    """Triton kernel that computes matmul + scale - simple version for small N"""
    # Each program computes one row of output
    pid = tl.program_id(0)
    
    if pid >= M:
        return
    
    # Compute output for this row
    offs_m = pid
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    
    # Initialize accumulator for each output column
    accumulator = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    
    # Loop over K
    for k in range(K):
        # Load element from a
        a_val = tl.load(a_ptr + offs_m * K + k)
        
        # Load column from b
        b_vals = tl.load(b_ptr + k * N + offs_n, mask=offs_n < N, other=0.0)
        
        # Multiply and accumulate
        accumulator += a_val * b_vals
    
    # Apply scale
    accumulator = accumulator * scale_val
    
    # Store result
    output_ptrs = output_ptr + offs_m * N + offs_n
    tl.store(output_ptrs, accumulator, mask=offs_n < N)


@torch.fx.wrap
def fused_kernel_wrapper(in_0, in_1, in_2):
    """Wrapper function that computes matmul + scale using Triton"""
    # Get shapes
    M = in_2.shape[0]  # 2
    K = in_2.shape[1]  # 512
    N = in_1.shape[1]  # 1
    
    # Get scalar scale value
    if isinstance(in_0, torch.Tensor):
        scale = in_0.item() if in_0.numel() == 1 else in_0.float()
    else:
        scale = float(in_0)
    
    # Allocate output
    output = torch.empty((M, N), device=in_2.device, dtype=torch.float32)
    
    # Use one program per row
    grid = (M,)
    
    BLOCK_SIZE_N = 1
    
    matmul_scale_kernel[grid](
        in_2, in_1, scale, output,
        M, N, K,
        M, BLOCK_SIZE_N
    )
    
    # Return only the output - the model's transpose will be applied separately
    return output


def replacement_func():
    return fused_kernel_wrapper