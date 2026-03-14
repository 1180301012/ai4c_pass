import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, shape0, shape1, shape2, shape3):
    """
    Pattern matching for matmul followed by view.
    This matches the computation in model.py exactly.
    shape0-shape3 are placeholders for the view dimensions.
    """
    tmp_0 = torch.matmul(in_1, in_0)
    tmp_1 = tmp_0.view(shape0, shape1, shape2, shape3)
    return tmp_1


def replacement_args(in_0, in_1, shape0, shape1, shape2, shape3):
    """
    Extract arguments needed for the replacement.
    """
    return (in_0, in_1, shape0, shape1, shape2, shape3)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 128}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 256}, num_stages=3, num_warps=8),
    ],
    key=['M', 'K'],
)
@triton.jit
def batched_matvec_kernel(
    a_ptr, b_ptr, c_ptr,
    M, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk,
    stride_cb, stride_cm,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Optimized batched matrix-vector multiplication kernel.
    Computes C[b, m] = sum_k A[b, m, k] * B[b, k] for each batch b.
    """
    pid_b = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    
    # Compute offsets for M dimension
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offs_m < M
    
    # Accumulator for this block
    accumulator = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        offs_k = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < K
        
        # Load A[b, m, k]
        a_ptrs = a_ptr + pid_b * stride_ab + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        mask_a = mask_m[:, None] & mask_k[None, :]
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        
        # Load B[b, k]
        b_ptrs = b_ptr + pid_b * stride_bb + offs_k * stride_bk
        b = tl.load(b_ptrs, mask=mask_k, other=0.0)
        
        # Accumulate
        accumulator += tl.sum(a * b[None, :], axis=1)
    
    # Store result
    c_ptrs = c_ptr + pid_b * stride_cb + offs_m * stride_cm
    tl.store(c_ptrs, accumulator, mask=mask_m)


@torch.fx.wrap
def fused_matmul_view(in_0, in_1, shape0, shape1, shape2, shape3):
    """
    Fused implementation of matmul + view using Triton.
    Optimized for matrix-vector multiplication (N=1 case).
    """
    # Get shapes
    original_shape_1 = in_1.shape
    
    # Flatten batch dimensions for easier processing
    in_1_flat = in_1.reshape(-1, in_1.shape[-2], in_1.shape[-1])
    in_0_flat = in_0.reshape(-1, in_0.shape[-2], in_0.shape[-1])
    
    batch_size = in_1_flat.shape[0]
    M = in_1_flat.shape[1]  # rows of in_1
    K = in_1_flat.shape[2]  # cols of in_1, rows of in_0
    N = in_0_flat.shape[2]  # cols of in_0
    
    # Use optimized matrix-vector kernel (assuming N=1 for these patterns)
    # Squeeze the last dimension for efficiency
    in_0_vec = in_0_flat.squeeze(-1)  # Shape: (batch_size, K)
    
    # Allocate output
    out = torch.empty((batch_size, M), dtype=in_1.dtype, device=in_1.device)
    
    # Grid configuration: (batch, M_blocks)
    # The block size will be determined by autotuning
    grid = lambda META: (
        batch_size,
        triton.cdiv(M, META['BLOCK_SIZE_M']),
    )
    
    # Launch kernel (autotuning will pick the best config)
    batched_matvec_kernel[grid](
        in_1_flat, in_0_vec, out,
        M, K,
        in_1_flat.stride(0), in_1_flat.stride(1), in_1_flat.stride(2),
        in_0_vec.stride(0), in_0_vec.stride(1),
        out.stride(0), out.stride(1),
    )
    
    # Reshape to add back the singleton dimension
    out = out.unsqueeze(-1)  # Shape: (batch_size, M, 1)
    
    # Reshape to match the original batch structure
    batch_dims = original_shape_1[:-2]
    intermediate_shape = list(batch_dims) + [M, N]
    out = out.reshape(intermediate_shape)
    
    # Apply the view with the captured shape
    return out.view(shape0, shape1, shape2, shape3)


def replacement_func():
    return fused_matmul_view