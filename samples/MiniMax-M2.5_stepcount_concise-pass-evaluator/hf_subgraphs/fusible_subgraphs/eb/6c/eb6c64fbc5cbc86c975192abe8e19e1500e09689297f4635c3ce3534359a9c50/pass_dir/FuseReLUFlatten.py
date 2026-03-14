import torch
import triton
import triton.language as tl


# Pattern matching function - matches F.relu
def pattern(x):
    # Match functional relu
    return torch.nn.functional.relu(x)


# Extract arguments from matched nodes
def replacement_args(x):
    return (x,)


# Optimized Triton kernel that applies ReLU and stores to flattened output
@triton.jit
def relu_flatten_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU: max(0, x)
    x = tl.where(x > 0.0, x, 0.0)
    
    # Store result
    tl.store(output_ptr + offsets, x, mask=mask)


@torch.fx.wrap
def relu_flatten_impl(x):
    # Use Triton kernel for ReLU
    B, C, H, W = x.shape
    n_elements = B * C * H * W
    
    # Create output tensor
    output = torch.empty_like(x)
    
    # Configure block size
    BLOCK_SIZE = 2048
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    relu_flatten_kernel[(num_programs,)](
        input_ptr=x,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Flatten the output
    output = output.flatten(1, -1)
    
    return output


def replacement_func():
    return relu_flatten_impl