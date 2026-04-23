import torch
import triton
import triton.language as tl

# Pattern matches: sigmoid + subtract 0.25 + multiply pi (fused)
# This receives the cat output from outside the matched subgraph
def pattern(in_5):
    tmp_5 = in_5.sigmoid()
    tmp_6 = tmp_5 - 0.25
    tmp_7 = tmp_6 * 3.141592653589793
    return tmp_7

def replacement_args(in_5):
    return (in_5,)

@triton.jit
def fused_sigmoid_sub_mul_kernel(
    in_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    
    # Compute sigmoid and fused sub/mul in one expression
    # sigmoid(x) * pi - pi/4 = (1 / (1 + exp(-x))) * pi - pi/4
    # Using tl.sigmoid directly (handles float32/64 natively)
    sigmoid_val = tl.sigmoid(x.to(tl.float32)).to(x.dtype)
    result = sigmoid_val * 3.141592653589793 - 0.7853981633974483
    
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def kernel_wrapper(in_5):
    n_elements = in_5.numel()
    out = torch.empty_like(in_5)
    
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Ensure at least one program
    num_programs = max(num_programs, 1)
    
    fused_sigmoid_sub_mul_kernel[(num_programs,)](
        in_ptr=in_5,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return kernel_wrapper