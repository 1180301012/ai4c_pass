import torch
import triton
import triton.language as tl
import math

def pattern(in_0):
    tmp_0 = torch.nn.functional.relu(in_0, inplace = True)
    tmp_1 = torch.sigmoid(tmp_0)
    return (tmp_1,)

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def fused_relu_sigmoid_kernel_bfloat16(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Fused computation: ReLU followed by Sigmoid
    relu_out = tl.maximum(x, 0.0)
    sigmoid_out = 1.0 / (1.0 + tl.exp(-relu_out))
    
    # Store result
    tl.store(out_ptr + offsets, sigmoid_out, mask=mask)

@torch.fx.wrap
def fused_relu_sigmoid_bfloat16(x):
    n_elements = x.numel()
    # Auto-tune BLOCK_SIZE or use a reasonable default
    BLOCK_SIZE = 1024
    num_programs = math.ceil(n_elements / BLOCK_SIZE)
    
    out = torch.empty_like(x)
    
    fused_relu_sigmoid_kernel_bfloat16[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_relu_sigmoid_bfloat16