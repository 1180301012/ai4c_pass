import torch
import triton
import triton.language as tl

# Pattern matching function - matches GELU + flatten
def pattern(in_0):
    tmp_0 = torch.nn.functional.gelu(in_0, approximate='none')
    tmp_1 = tmp_0.flatten(1, -1)
    return tmp_1

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Simple optimized kernel
@triton.jit
def gelu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    x = tl.load(x_ptr + i, mask=m)
    y = x * (0.5 + 0.5 * tl.math.erf(x * 0.7071067811865476))
    tl.store(out_ptr + i, y, mask=m)

@torch.fx.wrap  
def gelu_flatten_wrapper(x):
    b = x.shape[0]
    n = x.numel()
    f = n // b
    out = torch.empty((b, f), dtype=x.dtype, device=x.device)
    
    # Use single block for small tensors, multiple for larger
    if n <= 2048:
        BLOCK = 2048
        grid = (1,)
    else:
        BLOCK = 1024
        grid = ((n + BLOCK - 1) // BLOCK,)
    
    gelu_kernel[grid](x, out, n, BLOCK=BLOCK)
    return out

def replacement_func():
    return gelu_flatten_wrapper