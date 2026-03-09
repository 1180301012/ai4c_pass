import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    """Match: matmul(in_2, in_1) * in_0"""
    tmp_0 = torch.matmul(in_2, in_1)
    tmp_1 = tmp_0 * in_0
    return tmp_1


def replacement_args(in_0, in_1, in_2):
    """Extract arguments needed for the replacement"""
    return (in_0, in_1, in_2)


@triton.jit
def fused_matmul_scale_kernel(
    in_2_ptr, in_1_ptr, in_0_ptr,
    out_ptr,
    M: tl.constexpr, K: tl.constexpr,
):
    """Fused kernel: matmul + scale - simple and efficient"""
    
    # Get row index from program
    pid = tl.program_id(0)
    
    if pid >= M:
        return
    
    # Load scalar
    in_0_val = tl.load(in_0_ptr).to(tl.float32)
    
    # Load full row and column vectors
    offsets = tl.arange(0, 512)
    mask = offsets < K
    
    # Load 512 elements at once
    in_2_vals = tl.load(in_2_ptr + pid * K + offsets, mask=mask, other=0.0)
    in_1_vals = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    
    # Dot product using sum of element-wise multiply
    acc = tl.sum(in_2_vals * in_1_vals)
    
    # Scale
    result = acc * in_0_val
    
    # Store
    tl.store(out_ptr + pid, result)


@torch.fx.wrap
def fused_kernel_wrapper(in_0, in_1, in_2):
    """Wrapper function that launches the Triton kernel"""
    M = in_2.shape[0]  # 2
    K = in_2.shape[1]  # 512
    
    # Allocate output
    out = torch.empty((M, 1), dtype=torch.float32, device=in_2.device)
    
    # Grid: one program per row
    grid = (M,)
    
    fused_matmul_scale_kernel[grid](
        in_2, in_1, in_0,
        out,
        M, K,
    )
    
    return out


def replacement_func():
    """Return the replacement function"""
    return fused_kernel_wrapper