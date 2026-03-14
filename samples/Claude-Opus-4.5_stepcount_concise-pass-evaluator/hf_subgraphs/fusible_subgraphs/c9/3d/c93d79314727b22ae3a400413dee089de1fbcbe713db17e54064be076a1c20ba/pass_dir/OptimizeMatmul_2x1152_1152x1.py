import torch
import triton
import triton.language as tl


def pattern(a, b):
    """
    Pattern to match: torch.matmul(a, b)
    where a is [2, 1152] and b is [1152, 1]
    """
    result = torch.matmul(a, b)
    return result


def replacement_args(a, b):
    return (a, b)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_K': 64}, num_warps=1),
        triton.Config({'BLOCK_K': 128}, num_warps=1),
        triton.Config({'BLOCK_K': 256}, num_warps=2),
        triton.Config({'BLOCK_K': 512}, num_warps=4),
        triton.Config({'BLOCK_K': 1024}, num_warps=4),
        triton.Config({'BLOCK_K': 2048}, num_warps=8),
    ],
    key=['K'],
)
@triton.jit
def matmul_small_kernel(
    a_ptr, b_ptr, out_ptr,
    M, K, N,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_om, stride_on,
    BLOCK_K: tl.constexpr,
):
    """
    Optimized kernel for small matmul [M, K] @ [K, N]
    Each program computes one element of the output.
    """
    # Get the row and column index for this program
    row = tl.program_id(0)
    col = tl.program_id(1)
    
    # Initialize accumulator
    acc = tl.zeros((), dtype=tl.float32)
    
    # Compute dot product in chunks of BLOCK_K
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask = k_offsets < K
        
        # Load elements from a[row, k_offsets]
        a_ptrs = a_ptr + row * stride_am + k_offsets * stride_ak
        a_vals = tl.load(a_ptrs, mask=mask, other=0.0)
        
        # Load elements from b[k_offsets, col]
        b_ptrs = b_ptr + k_offsets * stride_bk + col * stride_bn
        b_vals = tl.load(b_ptrs, mask=mask, other=0.0)
        
        # Accumulate dot product
        acc += tl.sum(a_vals * b_vals)
    
    # Store the result
    out_ptr = out_ptr + row * stride_om + col * stride_on
    tl.store(out_ptr, acc)


@torch.fx.wrap
def triton_matmul_small(a, b):
    """
    Wrapper function for small matmul using Triton.
    """
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Inner dimensions must match: {K} vs {K2}"
    
    # Ensure contiguous tensors
    a = a.contiguous()
    b = b.contiguous()
    
    # Allocate output tensor
    out = torch.empty((M, N), dtype=a.dtype, device=a.device)
    
    # Launch kernel with grid = (M, N)
    grid = (M, N)
    
    matmul_small_kernel[grid](
        a, b, out,
        M, K, N,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
    )
    
    return out


def replacement_func():
    return triton_matmul_small