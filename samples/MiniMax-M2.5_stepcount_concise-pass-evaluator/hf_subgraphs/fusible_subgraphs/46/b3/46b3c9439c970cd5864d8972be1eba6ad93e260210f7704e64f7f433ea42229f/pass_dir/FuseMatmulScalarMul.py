import torch
import triton
import triton.language as tl


# Pattern matching function - match the exact computation
def pattern(in_0, in_1, in_2):
    """
    Match: matmul + multiply + transpose
    """
    tmp_0 = torch.matmul(in_2, in_1)
    tmp_1 = tmp_0 * in_0
    tmp_2 = tmp_1.t()
    return (tmp_1, tmp_2)


# Argument extraction function  
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# Triton kernel for matmul (the expensive part)
@triton.jit
def matmul_kernel(
    in_2_ptr, in_1_ptr, out_ptr,
    M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Matrix multiplication kernel: in_2 @ in_1"""
    row_idx = tl.program_id(0)
    col_idx = 0  # N=1
    
    acc = 0.0
    for k in range(0, K, BLOCK_SIZE):
        k_offsets = k + tl.arange(0, BLOCK_SIZE)
        mask_k = k_offsets < K
        
        in_2_ptrs = in_2_ptr + row_idx * K + k_offsets
        in_2_vals = tl.load(in_2_ptrs, mask=mask_k, other=0.0)
        
        in_1_ptrs = in_1_ptr + k_offsets * N
        in_1_vals = tl.load(in_1_ptrs, mask=mask_k, other=0.0)
        
        acc += tl.sum(in_2_vals * in_1_vals)
    
    tl.store(out_ptr + row_idx * N + col_idx, acc)


@torch.fx.wrap
def triton_matmul(in_1, in_2):
    """Triton wrapper for matmul"""
    M = in_2.shape[0]
    K = in_2.shape[1]
    N = in_1.shape[1]
    
    out = torch.empty((M, N), dtype=torch.float32, device=in_2.device)
    
    BLOCK_SIZE = 512
    grid = (M,)
    
    matmul_kernel[grid](in_2, in_1, out, M, K, N, BLOCK_SIZE)
    
    return out


# Replacement function that uses Triton for matmul, then does multiply and transpose in torch
def optimized_replacement(in_0, in_1, in_2):
    """Use Triton for matmul, then torch for the rest"""
    # Use Triton for the matmul
    tmp_0 = triton_matmul(in_1, in_2)
    # Scalar multiply (in_0 is a scalar)
    tmp_1 = tmp_0 * in_0
    # Transpose
    tmp_2 = tmp_1.t()
    return (tmp_1, tmp_2)


def replacement_func():
    return optimized_replacement