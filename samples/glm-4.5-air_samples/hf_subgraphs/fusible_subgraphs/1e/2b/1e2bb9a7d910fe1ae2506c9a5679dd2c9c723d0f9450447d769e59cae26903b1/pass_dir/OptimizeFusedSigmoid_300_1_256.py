import torch
import triton
import triton.language as tl

# Pattern matching function - matches two sigmoid operations
def pattern(x, y):
    # This matches the exact pattern from the model: two sigmoid operations
    # x and y are the input tensors to the sigmoid operations
    tmp_3 = x.sigmoid()
    tmp_4 = y.sigmoid()
    return (tmp_3, tmp_4)

# Argument extraction function  
def replacement_args(x, y):
    return (x, y)

# Optimized fused sigmoid kernel using Triton
@triton.jit
def fused_sigmoid_kernel(
    x1_ptr,
    x2_ptr,
    out1_ptr,
    out2_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values from both tensors
    x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0)
    x2 = tl.load(x2_ptr + offsets, mask=mask, other=0.0)
    
    # Apply sigmoid function: 1 / (1 + exp(-x))
    # Using numerically stable version
    out1 = tl.where(x1 >= 0,
                   1 / (1 + tl.exp(-x1)),
                   tl.exp(x1) / (1 + tl.exp(x1)))
    
    out2 = tl.where(x2 >= 0,
                   1 / (1 + tl.exp(-x2)),
                   tl.exp(x2) / (1 + tl.exp(x2)))
    
    # Store both results
    tl.store(out1_ptr + offsets, out1, mask=mask)
    tl.store(out2_ptr + offsets, out2, mask=mask)

# Kernel wrapper for fused sigmoid
@torch.fx.wrap
def optimized_fused_sigmoid(x1, x2):
    # Check that both inputs have the same shape
    if x1.shape != x2.shape:
        raise ValueError("Input tensors must have the same shape")
    
    n_elements = x1.numel()
    
    # Create output tensors
    out1 = torch.empty_like(x1)
    out2 = torch.empty_like(x2)
    
    # Choose optimal block size
    BLOCK_SIZE = 1024
    
    # Calculate number of programs needed
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_programs,)
    
    # Launch the fused kernel
    fused_sigmoid_kernel[grid](
        x1_ptr=x1,
        x2_ptr=x2,
        out1_ptr=out1,
        out2_ptr=out2,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out1, out2

# Replacement function
def replacement_func():
    return optimized_fused_sigmoid