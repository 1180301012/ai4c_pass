import torch
import triton
import triton.language as tl
import math

def pattern(x):
    # Match the exact structure: input -> ReLU -> used in view -> unsqueeze -> return both outputs
    relu_out = torch.nn.functional.relu(x, inplace=True)
    # Use the result in a view operation (like the original)
    viewed = relu_out.view(x.shape[0], 512, -1)
    unsqueezed = viewed.unsqueeze(1)
    # Return the same structure as original
    return unsqueezed, relu_out

def replacement_args(x):
    return (x,)

@triton.jit
def relu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute ReLU: max(0, x)
    out = tl.maximum(x, 0.0)
    
    # Store output
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_forward(x):
    # Apply optimized ReLU first
    n_elements = x.numel()
    
    # Use BLOCK_SIZE of 1024 for good GPU occupancy
    BLOCK_SIZE = 1024
    num_programs = math.ceil(n_elements / BLOCK_SIZE)
    
    # Create output tensor for ReLU result
    relu_out = torch.empty_like(x)
    
    # Launch ReLU kernel
    relu_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=relu_out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Apply view and unsqueeze operations (these are cheap, just metadata)
    viewed = relu_out.view(x.shape[0], 512, -1)
    unsqueezed = viewed.unsqueeze(1)
    
    # Return both values to match the original structure
    return unsqueezed, relu_out

def replacement_func():
    return optimized_forward