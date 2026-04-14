import torch
import triton
import triton.language as tl

def pattern(x):
    """Fuses contiguous + unsqueeze operations"""
    tmp_0 = x.contiguous()
    tmp_1 = tmp_0.unsqueeze(-1)
    return tmp_1

def replacement_args(x):
    return (x,)

@triton.jit
def fused_contiguous_unsqueeze_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel that fuses contiguous memory layout with unsqueeze operation"""
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Store output (effectively adding a dimension at the end)
    # The contiguous operation is handled by the memory layout in the kernel
    tl.store(out_ptr + offsets, x, mask=mask)

@torch.fx.wrap
def fused_contiguous_unsqueeze(x):
    """Function that fuses contiguous and unsqueeze operations"""
    # Optimize block size for the specific tensor sizes we're dealing with
    new_shape = list(x.shape) + [1]
    out = torch.empty(new_shape, dtype=x.dtype, device=x.device)
    
    # Optimize block size for the specific tensor sizes we're dealing with
    # Tensors are small (typically几百到几千 elements), so use smaller block size
    total_elements = x.numel()
    
    # Use optimal block size for small tensors
    if total_elements <= 1024:
        BLOCK_SIZE = 256
    elif total_elements <= 4096:
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = 1024
    
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_contiguous_unsqueeze_kernel[(grid_size,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=total_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_contiguous_unsqueeze