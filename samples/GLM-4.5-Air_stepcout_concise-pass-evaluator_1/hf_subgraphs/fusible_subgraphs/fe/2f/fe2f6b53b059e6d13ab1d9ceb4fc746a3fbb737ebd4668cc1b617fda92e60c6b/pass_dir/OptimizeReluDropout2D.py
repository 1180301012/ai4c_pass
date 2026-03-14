import torch
import triton
import triton.language as tl
import math

def pattern(x):
    tmp_0 = torch.nn.functional.relu(x, inplace=True)
    tmp_1 = torch.nn.functional.dropout2d(tmp_0, 0.1, False, False)
    return (tmp_1, tmp_0)

def replacement_args(x):
    return (x,)

@triton.jit
def relu_dropout2d_kernel(
    x_ptr,
    relu_out_ptr,
    dropout_out_ptr,
    n_elements,
    dropout_p: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # ReLU: max(0, x)
    relu_out = tl.maximum(x, 0.0)
    
    # Dropout2D: Since dropout2d is called with training=False,
    # it should return the input unchanged (no dropout applied in eval mode)
    dropout_out = relu_out
    
    # Store results
    tl.store(relu_out_ptr + offsets, relu_out, mask=mask)
    tl.store(dropout_out_ptr + offsets, dropout_out, mask=mask)

@torch.fx.wrap
def optimized_relu_dropout2d(x):
    # Get tensor properties
    n_elements = x.numel()
    
    # Set block size - optimize for typical GPU architecture
    BLOCK_SIZE = 512
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensors
    relu_out = torch.empty_like(x)
    dropout_out = torch.empty_like(x)
    
    # Launch kernel
    relu_dropout2d_kernel[(num_programs,)](
        x_ptr=x,
        relu_out_ptr=relu_out,
        dropout_out_ptr=dropout_out,
        n_elements=n_elements,
        dropout_p=0.1,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return (dropout_out, relu_out)

def replacement_func():
    return optimized_relu_dropout2d