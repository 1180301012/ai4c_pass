import torch
import triton
import triton.language as tl

# Pattern matching function - exactly match the model structure
# Uses nested call structure like the demo example
def pattern(in_0):
    # Match both relu and dropout2d with their exact parameters
    # The model uses inplace=True for relu and training=False for dropout2d
    return torch.nn.functional.dropout2d(
        torch.nn.functional.relu(in_0, inplace=True), 
        0.1, 
        False,  # training=False
        False   # inplace=False
    )

# Extract arguments needed for replacement
def replacement_args(in_0):
    return (in_0,)

# Optimized Triton kernel for fused ReLU + Dropout2d (inference mode)
@triton.jit
def fused_relu_dropout_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU: max(0, x)
    x = tl.where(x > 0, x, 0.0)
    
    # Dropout2d with training=False is a no-op, just store the result
    tl.store(output_ptr + offsets, x, mask=mask)


@torch.fx.wrap
def triton_fused_relu_dropout(in_0):
    # Get total number of elements
    n_elements = in_0.numel()
    
    # Determine BLOCK_SIZE
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty_like(in_0)
    
    # Launch kernel
    fused_relu_dropout_kernel[(num_programs,)](
        input_ptr=in_0,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return triton_fused_relu_dropout