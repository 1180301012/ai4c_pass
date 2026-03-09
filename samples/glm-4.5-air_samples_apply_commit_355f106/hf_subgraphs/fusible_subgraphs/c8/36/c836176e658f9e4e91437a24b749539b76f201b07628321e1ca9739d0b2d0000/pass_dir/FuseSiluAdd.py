import torch
import triton
import triton.language as tl

def pattern(in_1, in_0):
    return torch.nn.functional.silu(in_1, inplace=True) + in_0

def replacement_args(in_1, in_0):
    return (in_1, in_0)

@triton.jit
def fused_silu_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Fused SILU + Add: out = (x * sigmoid(x)) + y
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    out = (x * sigmoid_x) + y
    
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_fused_silu_add(in_1, in_0):
    N = in_1.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(in_1)
    
    fused_silu_add_kernel[(num_programs,)](
        x_ptr=in_1,
        y_ptr=in_0,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return triton_fused_silu_add