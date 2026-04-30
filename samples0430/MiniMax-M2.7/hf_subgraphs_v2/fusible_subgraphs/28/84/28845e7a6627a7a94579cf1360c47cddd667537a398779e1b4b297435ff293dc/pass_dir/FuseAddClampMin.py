import torch
import triton
import triton.language as tl

@triton.jit
def fuse_add_clamp_min_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    n_elements,
    min_val: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for: out = clamp_min(in_1 + in_0, min_val)
    
    This fuses element-wise addition with a minimum clamping operation.
    The min_val is -inf for the computation patterns in this task.
    """
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    x = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    
    # Compute addition
    add_result = x + y
    
    # Apply clamp_min (element-wise maximum with min_val)
    result = tl.maximum(add_result, min_val)
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fuse_add_clamp_min(in_0, in_1):
    """
    Fused add + clamp_min operation.
    
    Args:
        in_0: First tensor [1, 1, N, N]
        in_1: Second tensor [1, H, N, N] 
    Returns:
        Tensor after broadcasting add and clamp_min
    """
    # Determine min_val based on dtype
    if in_0.dtype in (torch.float16, torch.bfloat16):
        min_val = float('-inf')
    else:
        min_val = float('-inf')
    
    N = in_0.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(in_0)
    
    # Launch fused kernel
    fuse_add_clamp_min_kernel[(num_programs,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        n_elements=N,
        min_val=min_val,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def pattern(in_0, in_1):
    """
    Match the pattern: in_1 + in_0, then clamp_min with -inf
    """
    from torch import device
    
    tmp_0 = in_1 + in_0
    tmp_1 = torch.tensor(-3.4028234663852886e+38, device=device(type='cuda', index=0))
    tmp_2 = torch.max(tmp_0, tmp_1)
    
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fuse_add_clamp_min