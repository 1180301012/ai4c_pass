import torch
import triton
import triton.language as tl

def pattern(in_2):
    """
    Match standalone sigmoid operation on input tensor with shape [300, 1, 256]
    """
    tmp_3 = in_2.sigmoid()
    return tmp_3

def replacement_args(in_2):
    return (in_2,)

@triton.jit
def optimized_sigmoid_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensor
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate sigmoid directly using exponential function
    # sigmoid(x) = 1 / (1 + exp(-x))
    neg_x = -x
    exp_neg_x = tl.exp(neg_x)
    out = 1.0 / (1.0 + exp_neg_x)
    
    # Store result
    tl.store(y_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_sigmoid(in_2):
    N = in_2.numel()
    
    # Use the optimal block size based on input size
    # For 76,800 elements, use largest block size to maximize GPU occupancy
    if N < 1000:
        BLOCK_SIZE = 256
    elif N < 10000:
        BLOCK_SIZE = 1024
    elif N < 50000:
        BLOCK_SIZE = 2048
    else:
        BLOCK_SIZE = 8192  # Use maximum block size for large inputs
    
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(in_2)
    
    optimized_sigmoid_kernel[(num_programs,)](
        in_2,
        out,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_sigmoid