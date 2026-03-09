import torch
import triton
import triton.language as tl

# Pattern matching function - matches the entire computation
def pattern(mod, in_0):
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = tmp_0.view(mod.batch_size, mod.channels, mod.height * mod.width)
    tmp_2 = tmp_1.unsqueeze(1)
    return tmp_2, tmp_0  # Return both values that the original function returns

# Argument extraction function
def replacement_args(mod, in_0):
    return (in_0, mod.batch_size, mod.channels, mod.height * mod.width)

# Optimized ReLU kernel
@triton.jit
def relu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU: max(0, x)
    out = tl.maximum(x, 0.0)
    
    # Store result (can be in-place if we reuse the same memory)
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_forward(in_0, batch_size, channels, flattened_dim):
    # Optimized ReLU + View + Unsqueeze combined operation
    # Handle in-place ReLU first
    if in_0.is_contiguous():
        # For in-place operation, we operate directly on the input tensor
        N = in_0.numel()
        BLOCK_SIZE = 1024
        num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        # Launch kernel with output being the same as input (in-place)
        relu_kernel[(num_programs,)](
            x_ptr=in_0,
            out_ptr=in_0,
            n_elements=N,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # If not contiguous, we need to create a new tensor
        N = in_0.numel()
        BLOCK_SIZE = 1024
        num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        out = torch.empty_like(in_0)
        relu_kernel[(num_programs,)](
            x_ptr=in_0,
            out_ptr=out,
            n_elements=N,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        in_0 = out
    
    # Continue with view and unsqueeze operations
    tmp_1 = in_0.view(batch_size, channels, flattened_dim)
    tmp_2 = tmp_1.unsqueeze(1)
    
    return tmp_2, in_0  # Return both tensors

# Replacement function
def replacement_func():
    return optimized_forward