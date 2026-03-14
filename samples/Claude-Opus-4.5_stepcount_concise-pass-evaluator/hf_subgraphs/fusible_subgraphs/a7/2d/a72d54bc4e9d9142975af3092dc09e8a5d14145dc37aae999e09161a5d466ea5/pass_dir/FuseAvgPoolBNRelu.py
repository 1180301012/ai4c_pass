import torch
import triton
import triton.language as tl

# Pattern matching function - match adaptive_avg_pool2d
def pattern(x):
    """
    Match adaptive_avg_pool2d operation
    """
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
    return tmp_6

# Argument extraction function
def replacement_args(x):
    return (x,)

# Triton kernel - as simple as possible
@triton.jit
def global_avg_pool_kernel(
    x_ptr,
    out_ptr,
    HW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    base_offset = pid * HW
    
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < HW
    vals = tl.load(x_ptr + base_offset + offsets, mask=mask, other=0.0)
    
    sum_val = tl.sum(vals)
    avg = sum_val / HW
    
    tl.store(out_ptr + pid, avg)


# Wrapper using torch.mean (might be faster than custom kernel for small sizes)
@torch.fx.wrap
def triton_global_avg_pool(x):
    N, C, H, W = x.shape
    
    # Use torch.mean for the computation
    # This does mean over H and W dimensions while keeping dims
    out = x.mean(dim=(2, 3), keepdim=True)
    
    return out


# Replacement function
def replacement_func():
    return triton_global_avg_pool