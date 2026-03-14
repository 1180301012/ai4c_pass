import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
    return tmp_0

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def silu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # SILU: x * sigmoid(x) = x / (1 + exp(-x))
    out = x / (1.0 + tl.exp(-x))
    
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_silu(x):
    if x.numel() == 0:
        return torch.empty_like(x)
    
    # Use BLOCK_SIZE that's efficient for the tensor sizes we have
    BLOCK_SIZE = 1024
    
    # Reshape 2D/4D tensors to 1D for easier processing
    if x.dim() > 1:
        original_shape = x.shape
        x_1d = x.flatten()
    else:
        x_1d = x
    
    n_elements = x_1d.numel()
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out_1d = torch.empty_like(x_1d)
    
    silu_kernel[(num_programs,)](
        x_ptr=x_1d,
        out_ptr=out_1d,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape back to original dimensions
    if x.dim() > 1:
        return out_1d.reshape(original_shape)
    return out_1d

def replacement_func():
    return triton_silu