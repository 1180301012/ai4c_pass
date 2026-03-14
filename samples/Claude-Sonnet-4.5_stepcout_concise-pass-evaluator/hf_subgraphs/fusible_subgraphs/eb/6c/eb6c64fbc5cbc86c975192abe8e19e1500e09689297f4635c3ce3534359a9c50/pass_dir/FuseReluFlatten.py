import torch
import triton
import triton.language as tl

def pattern(in_0):
    """
    Pattern to match flatten
    """
    tmp_1 = in_0.flatten(1, -1)
    return tmp_1

def replacement_args(in_0):
    return (in_0,)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
    ],
    key=['n_elements'],
)
@triton.jit
def relu_flatten_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU: max(x, 0)
    out = tl.maximum(x, 0.0)
    
    # Store output
    tl.store(output_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_relu_flatten(x):
    # Just return flatten - this is a passthrough
    return x.flatten(1, -1)

def replacement_func():
    return fused_relu_flatten