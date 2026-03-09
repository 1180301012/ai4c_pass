import torch
import triton
import triton.language as tl

def pattern(x, module=None):
    """Pattern matching ReLU -> Dropout(0.0) -> Flatten sequence"""
    relu_out = torch.nn.functional.relu(x, inplace=False)
    dropout_out = torch.nn.functional.dropout(relu_out, 0.0, False, False)  # dropout rate=0.0 is no-op
    flatten_out = dropout_out.flatten(1, -1)
    return flatten_out

def replacement_args(x, module=None):
    return (x,)

@triton.jit
def optimized_relu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Optimized ReLU kernel without dropout overhead"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.maximum(x, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_relu_flatten(x):
    """Optimized function that combines ReLU with flattening, eliminating no-op dropout"""
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    optimized_relu_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out.flatten(1, -1)

def replacement_func():
    return optimized_relu_flatten