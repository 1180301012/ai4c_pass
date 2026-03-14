import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0):
    """
    Match ReLU (inplace=True) followed by Dropout2d (training=False)
    This pattern appears in the model computation:
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = torch.nn.functional.dropout2d(tmp_0, 0.1, False, False)
    return (tmp_1, tmp_0)
    """
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = torch.nn.functional.dropout2d(tmp_0, 0.1, False, False)
    return (tmp_1, tmp_0)

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Optimized kernel using Triton
@triton.jit
def fused_relu_dropout_kernel(
    x_ptr,
    tmp_0_ptr,
    tmp_1_ptr,
    N,  # Number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU activation
    relu_out = tl.maximum(x, 0.0)
    
    # Since training=False, dropout is just identity operation
    # So both tmp_0 and tmp_1 are the same as relu_out
    dropout_out = relu_out
    
    # Store results: both tmp_0 and tmp_1 get the same value
    tl.store(tmp_0_ptr + offsets, relu_out, mask=mask)
    tl.store(tmp_1_ptr + offsets, dropout_out, mask=mask)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_relu_dropout_forward(in_0):
    # Get tensor shape and compute total elements
    N = in_0.numel()
    
    # Create output tensors (both should be the same as ReLU output)
    tmp_0 = torch.empty_like(in_0)
    tmp_1 = torch.empty_like(in_0)
    
    # Optimize block size based on tensor size
    if N <= 1024:
        BLOCK_SIZE = 1024
    elif N <= 8192:
        BLOCK_SIZE = 4096
    else:
        BLOCK_SIZE = 8192
    
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the fused kernel
    fused_relu_dropout_kernel[(num_programs,)](
        x_ptr=in_0,
        tmp_0_ptr=tmp_0,
        tmp_1_ptr=tmp_1,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return (tmp_1, tmp_0)

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_relu_dropout_forward