import torch
import triton
import triton.language as tl

@triton.jit
def fused_sigmoid_expand_add_relu_kernel(
    x_ptr,                      # in_0: [1, 2048, H, W]
    y_ptr,                      # in_1: [1, 2048, H, W] 
    z_ptr,                      # in_2: [1, 1, 2048]
    out_ptr,                    # [1, 2048, H, W]
    n_elements,                 # total elements in output
    H: tl.constexpr,            # spatial height (16 or 12)
    W: tl.constexpr,            # spatial width (16 or 12)
    C: tl.constexpr,            # channels (2048)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Convert linear offset to 4D coordinates
    offset = offsets
    h = offset % H
    offset = offset // H
    w = offset % W  
    offset = offset // W
    c = offset % C
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)              # in_0
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)              # in_1
    
    # Load sigmoid weights (z has shape [1, 1, 2048])
    z_weight = tl.load(z_ptr + c, mask=c < C, other=0.0)
    
    # Compute fused operations:
    # sigmoid(z) -> view(1, -1, 1, 1) -> expand -> (y * expanded_sigmoid) + x -> relu
    sigmoid_val = 1.0 / (1.0 + tl.exp(-z_weight))
    fused_val = y * sigmoid_val + x
    relu_val = tl.maximum(fused_val, 0.0)
    
    # Store result
    tl.store(out_ptr + offsets, relu_val, mask=mask)

@torch.fx.wrap
def simple_sigmoid(x):
    """Simple optimized sigmoid implementation using Triton"""
    # Use Triton to compute sigmoid
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty_like(x)
    
    # Define a simple sigmoid kernel
    @triton.jit
    def sigmoid_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        block_start = tl.program_id(0) * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        out = 1.0 / (1.0 + tl.exp(-x))
        tl.store(out_ptr + offsets, out, mask=mask)
    
    sigmoid_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def pattern(in_0, in_1, in_2):
    """Simple test pattern - just sigmoid"""
    result = in_2.sigmoid()
    return result

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

def replacement_func():
    return simple_sigmoid