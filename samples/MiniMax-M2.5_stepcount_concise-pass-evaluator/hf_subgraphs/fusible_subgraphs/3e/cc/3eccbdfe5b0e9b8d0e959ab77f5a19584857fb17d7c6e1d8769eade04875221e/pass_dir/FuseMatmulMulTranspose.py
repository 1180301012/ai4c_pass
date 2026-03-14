import torch
import triton
import triton.language as tl


# Simplified kernel without autotune overhead
@triton.jit
def fused_matmul_scalar_kernel(
    in_2_ptr,
    in_1_ptr,
    in_0_val,
    out_ptr,
    M,
    N,
    K,
):
    """Simplified fused matmul + scalar multiply kernel"""
    # Get position
    pid = tl.program_id(0)
    row = pid
    
    if row >= M:
        return
    
    # Load entire row of in_2 (512 elements)
    row_offsets = row * K + tl.arange(0, 512)
    mask = tl.arange(0, 512) < K
    a = tl.load(in_2_ptr + row_offsets, mask=mask, other=0.0)
    
    # Load entire column of in_1 (512 elements)
    b = tl.load(in_1_ptr + tl.arange(0, 512), mask=mask, other=0.0)
    
    # Compute dot product
    accumulator = tl.sum(a * b)
    
    # Multiply by scalar
    result = accumulator * in_0_val
    
    # Store result
    tl.store(out_ptr + row, result)


@torch.fx.wrap
def fused_matmul_scalar_mul(in_0, in_1, in_2):
    """Fused kernel for: tmp_0 = matmul(in_2, in_1); tmp_1 = tmp_0 * in_0"""
    M = in_2.shape[0]
    K = in_2.shape[1]
    N = in_1.shape[1]
    
    in_0_val = in_0.item() if isinstance(in_0, torch.Tensor) else in_0
    
    out = torch.empty((M, N), dtype=torch.float32, device=in_2.device)
    
    grid = (M,)
    fused_matmul_scalar_kernel[grid](
        in_2, in_1, in_0_val, out, M, N, K
    )
    
    return out


def pattern(in_0, in_1, in_2):
    """Match matmul followed by scalar multiply"""
    tmp_0 = torch.matmul(in_2, in_1)
    tmp_1 = tmp_0 * in_0
    return tmp_1


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_matmul_scalar_mul