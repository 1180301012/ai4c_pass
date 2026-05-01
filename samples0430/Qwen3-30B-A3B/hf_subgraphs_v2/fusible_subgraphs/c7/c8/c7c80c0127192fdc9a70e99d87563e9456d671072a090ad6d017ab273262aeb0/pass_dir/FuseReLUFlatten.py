import torch
import triton
import triton.language as tl

def pattern(x):
    tmp_0 = torch.nn.functional.relu(x)
    tmp_1 = tmp_0.flatten(1, -1)
    return tmp_1

def replacement_args(x):
    return (x,)

@triton.jit
def fused_relu_flatten_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    y = tl.maximum(x, 0.0)
    
    tl.store(out_ptr + offsets, y, mask=mask)

@torch.fx.wrap
def fused_relu_flatten(x):
    B, C, H, W = x.shape
    assert H == 1 and W == 1, "Expected spatial dimensions 1,1"
    n_elements = B * C
    
    out = torch.empty((B, C), dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE = 128
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_relu_flatten_kernel[(num_blocks,)](
        in_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_relu_flatten