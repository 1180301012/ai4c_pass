import torch
import triton
import triton.language as tl

# Pattern 3: Unsqueeze operation
def pattern(in_tensor):
    tmp_13 = in_tensor.unsqueeze(-2)
    return tmp_13

def replacement_args(in_tensor):
    return (in_tensor,)

@triton.jit
def unsqueeze_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Store output data (unsqueeze adds a dimension but keeps the same data)
    tl.store(output_ptr + offsets, input_data, mask=mask)

@torch.fx.wrap
def optimized_unsqueeze(in_tensor):
    # For unsqueeze operation, we need to understand the shape transformation
    # input shape: [batch_size, features] -> output shape: [batch_size, 1, features]
    input_shape = in_tensor.shape
    batch_size = input_shape[0]
    features = input_shape[1]
    
    # Create output with appropriate shape
    output_shape = (batch_size, 1, features)
    output = torch.empty(output_shape, device=in_tensor.device, dtype=torch.float32)
    
    # Since unsqueeze(-2) just adds a dimension at position -2,
    # we need to copy the data from input to output appropriately
    # Input: [batch_size, features] -> Output: [batch_size, 1, features]
    
    # For each batch item, copy features to the new dimension
    N = batch_size * features
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    unsqueeze_kernel[(num_programs,)](
        in_tensor,
        output,
        N,
        BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_unsqueeze