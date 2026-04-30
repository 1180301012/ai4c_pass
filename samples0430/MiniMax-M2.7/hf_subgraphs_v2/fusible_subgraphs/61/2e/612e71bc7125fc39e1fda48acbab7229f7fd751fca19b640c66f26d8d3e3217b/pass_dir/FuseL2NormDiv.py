import torch
import triton
import triton.language as tl


@triton.jit
def fused_norm_div_kernel(
    in_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for L2 normalization: out = in / ||in||_2
    
    This kernel computes the L2 norm along the last dimension (dim=-1)
    and divides each element by its row norm.
    Uses a single-pass approach with early exit for small tensors.
    """
    # Compute row position
    row_idx = tl.program_id(0)
    
    # Early exit for small N values - use serial computation
    if N <= 512:
        # For small tensors, compute serially
        sum_sq = 0.0
        for i in range(N):
            val = tl.load(in_ptr + row_idx * N + i).to(tl.float32)
            sum_sq += val * val
        norm = tl.sqrt(sum_sq + 1e-12)
        
        for i in range(N):
            val = tl.load(in_ptr + row_idx * N + i).to(tl.float32)
            normalized = (val / norm).to(tl.bfloat16)
            tl.store(out_ptr + row_idx * N + i, normalized)
    else:
        # For larger tensors, use parallel computation
        col_offsets = tl.arange(0, BLOCK_SIZE)
        sum_sq = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        
        for n in range(0, N, BLOCK_SIZE):
            mask = (col_offsets + n) < N
            a = tl.load(in_ptr + row_idx * N + col_offsets + n, mask=mask, other=0.0)
            sum_sq += tl.where(mask, a * a, 0.0)
        
        sum_sq = tl.sum(sum_sq, axis=0)
        norm = tl.sqrt(sum_sq + 1e-12)
        
        for n in range(0, N, BLOCK_SIZE):
            col_offsets = tl.arange(0, BLOCK_SIZE)
            mask = (col_offsets + n) < N
            a = tl.load(in_ptr + row_idx * N + col_offsets + n, mask=mask, other=0.0)
            normalized = (a / norm).to(tl.bfloat16)
            tl.store(out_ptr + row_idx * N + col_offsets + n, normalized, mask=mask)


@torch.fx.wrap
def triton_l2_norm_div(in_1):
    """
    Optimized L2 normalization kernel.
    
    Args:
        in_1: Input tensor of shape [M, N] to normalize
    
    Returns:
        Normalized tensor of shape [M, N]
    """
    M, N = in_1.shape
    
    # Ensure BLOCK_SIZE is a power of 2 and at most 1024
    BLOCK_SIZE = 1024
    while BLOCK_SIZE > N:
        BLOCK_SIZE //= 2
    if BLOCK_SIZE < 1:
        BLOCK_SIZE = 1
    
    # Create output tensor
    out = torch.empty_like(in_1)
    
    # Launch kernel - one block per row
    grid = (M,)
    
    fused_norm_div_kernel[grid](
        in_ptr=in_1,
        out_ptr=out,
        M=M,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def pattern(in_1):
    """
    Match the L2 normalization pattern: in_1 / norm(in_1, p=2, dim=-1, keepdim=True)
    This pattern appears in all three graphs.
    """
    tmp_0 = in_1.norm(p=2, dim=-1, keepdim=True)
    tmp_1 = in_1 / tmp_0
    return tmp_1


def replacement_args(in_1):
    return (in_1,)


def replacement_func():
    return triton_l2_norm_div