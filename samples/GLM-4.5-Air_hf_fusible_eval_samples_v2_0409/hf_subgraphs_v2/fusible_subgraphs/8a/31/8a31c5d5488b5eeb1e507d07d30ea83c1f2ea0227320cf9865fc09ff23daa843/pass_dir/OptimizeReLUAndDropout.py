import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = torch.nn.functional.dropout2d(tmp_0, 0.1, False, False)
    return (tmp_1, tmp_0)

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def relu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    out = tl.maximum(x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)

@triton.jit
def dropout_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    p: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    scale = 1.0 / (1.0 - p)
    out = x * scale
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_relu(x):
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    relu_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

@torch.fx.wrap
def optimized_dropout(x, p=0.1):
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    dropout_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        p=p,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

@torch.fx.wrap
def fused_relu_dropout(in_0, p=0.1):
    # Step 1: Optimized ReLU
    tmp_0 = optimized_relu(in_0)
    
    # Step 2: Optimized Dropout (training=False just scales by 1/(1-p))
    tmp_1 = optimized_dropout(tmp_0, p)
    
    return (tmp_1, tmp_0)

def replacement_func():
    return fused_relu_dropout