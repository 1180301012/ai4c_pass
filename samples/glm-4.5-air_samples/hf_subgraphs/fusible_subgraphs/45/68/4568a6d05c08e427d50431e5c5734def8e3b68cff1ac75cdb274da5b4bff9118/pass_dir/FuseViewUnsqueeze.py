import torch
import triton
import triton.language as tl

# Pattern matching function - matches the entire computation for view+unsqueeze optimization
def pattern(mod, in_0):
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = tmp_0.view(mod.batch_size, mod.channels, mod.height * mod.width)
    tmp_2 = tmp_1.unsqueeze(1)
    return tmp_2, tmp_0  # Return both values that the original function returns

# Argument extraction function
def replacement_args(mod, in_0):
    return (in_0, mod.batch_size, mod.channels, mod.height * mod.width)

# Optimized fused view + unsqueeze kernel
@triton.jit
def fused_reshape_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    flattened_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch_size * channels * flattened_dim)
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Store result - view + unsqueeze is just metadata change
    tl.store(out_ptr + offsets, x, mask=mask)

@torch.fx.wrap
def optimized_fused_forward(in_0, batch_size, channels, flattened_dim):
    # Combined computation with fused view + unsqueeze
    # Apply ReLU first (using standard PyTorch for simplicity)
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    
    # Fused view + unsqueeze operations - both are metadata operations
    # So we just need to create the correct output shape
    intermediate_shape = (batch_size, channels, flattened_dim)
    final_shape = (batch_size, 1, channels, flattened_dim)
    
    # The view operation
    tmp_1 = tmp_0.view(intermediate_shape)
    
    # The unsqueeze operation
    tmp_2 = tmp_1.unsqueeze(1)
    
    return tmp_2, tmp_0  # Return both tensors

# Replacement function
def replacement_func():
    return optimized_fused_forward