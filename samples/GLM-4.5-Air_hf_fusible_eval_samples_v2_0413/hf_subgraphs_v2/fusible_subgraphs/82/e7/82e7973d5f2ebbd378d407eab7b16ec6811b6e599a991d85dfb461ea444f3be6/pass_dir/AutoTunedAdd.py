import torch
import triton
import triton.language as tl

# Pattern matching function for addition
def pattern(in_0, in_1):
    """Simple addition pattern with autotuning"""
    tmp_1 = in_0 + in_1
    return (tmp_1,)

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Autotuned kernel that automatically finds best configuration
@triton.autotune(
    configs=[
        triton.Config(block_size=256, num_warps=1, num_stages=1),
        triton.Config(block_size=512, num_warps=1, num_stages=1),
        triton.Config(block_size=1024, num_warps=1, num_stages=1),
        triton.Config(block_size=2048, num_warps=1, num_stages=1),
        triton.Config(block_size=256, num_warps=2, num_stages=1),
        triton.Config(block_size=512, num_warps=2, num_stages=1),
        triton.Config(block_size=1024, num_warps=2, num_stages=1),
        triton.Config(block_size=2048, num_warps=2, num_stages=1),
        triton.Config(block_size=256, num_warps=4, num_stages=1),
        triton.Config(block_size=512, num_warps=4, num_stages=1),
        triton.Config(block_size=1024, num_warps=4, num_stages=1),
        triton.Config(block_size=2048, num_warps=4, num_stages=1),
    ],
    key=['n_elements'],
)
@triton.jit
def autotuned_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    block_size: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x + y
    
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def autotuned_add(in_0, in_1):
    """Autotuned addition operation"""
    n_elements = in_0.numel()
    
    # Use autotuned kernel
    num_programs = (n_elements + 2048 - 1) // 2048  # Conservative grid size
    
    out = torch.empty_like(in_0)
    
    if n_elements > 0:
        autotuned_add_kernel[(num_programs,)](
            x_ptr=in_0,
            y_ptr=in_1,
            out_ptr=out,
            n_elements=n_elements,
        )
    
    return out

# Replacement function
def replacement_func():
    return autotuned_add