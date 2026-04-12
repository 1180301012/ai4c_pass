import torch
import triton
import triton.language as tl
from torch import device

# Pattern matching function - matches the element-wise addition
def pattern(input_0, indexed_weights):
    result = input_0 + indexed_weights
    return result

# Argument extraction function
def replacement_args(input_0, indexed_weights):
    return (input_0, indexed_weights)

# Optimized Triton kernel for element-wise addition with broadcasting
@triton.jit
def optimized_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    x_stride_0,
    x_stride_1,
    x_stride_2,
    y_stride_0,
    y_stride_1,
    y_stride_2,
    out_stride_0,
    out_stride_1,
    out_stride_2,
    batch_size,
    seq_len,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one hidden dimension position
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_size
    
    # Get batch and sequence indices
    batch_offset = tl.program_id(1)
    seq_offset = tl.program_id(2)
    
    # Calculate base addresses
    x_base = batch_offset * x_stride_0 + seq_offset * x_stride_1
    y_base = batch_offset * y_stride_0 + seq_offset * y_stride_1
    out_base = batch_offset * out_stride_0 + seq_offset * out_stride_1
    
    # Load data (each thread handles a hidden dimension position)
    x_data = tl.load(x_ptr + x_base + offsets * x_stride_2, mask=mask, other=0.0)
    y_data = tl.load(y_ptr + y_base + offsets * y_stride_2, mask=mask, other=0.0)
    
    # Perform addition
    out_data = x_data + y_data
    
    # Store result
    tl.store(out_ptr + out_base + offsets * out_stride_2, out_data, mask=mask)

# Kernel wrapper for optimized operation
@torch.fx.wrap
def optimized_add_elementwise(input_0, indexed_weights):
    input_shape = input_0.shape
    # Assuming indexed_weights is already in shape (1, 9, hidden_size)
    weights_shape = indexed_weights.shape
    
    batch_size, seq_len, hidden_size = input_shape
    
    BLOCK_SIZE = 1024
    num_blocks = (hidden_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(input_0)
    
    optimized_add_kernel[
        (num_blocks, batch_size, seq_len)
    ](
        x_ptr=input_0,
        y_ptr=indexed_weights,
        out_ptr=output,
        x_stride_0=input_shape[1] * input_shape[2],
        x_stride_1=input_shape[2],
        x_stride_2=1,
        y_stride_0=weights_shape[1] * weights_shape[2],
        y_stride_1=weights_shape[2],
        y_stride_2=1,
        out_stride_0=input_shape[1] * input_shape[2],
        out_stride_1=input_shape[2],
        out_stride_2=1,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function (returns function reference)
def replacement_func():
    return optimized_add_elementwise