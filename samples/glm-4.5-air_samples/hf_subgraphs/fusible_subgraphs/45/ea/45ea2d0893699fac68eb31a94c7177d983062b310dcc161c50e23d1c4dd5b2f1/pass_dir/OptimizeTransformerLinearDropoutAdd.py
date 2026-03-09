import torch
import triton
import triton.language as tl

def pattern(bias, weight, x, addend):
    # Linear transformation: y = x @ weight.t() + bias
    linear_out = torch.nn.functional.linear(x, weight, bias)
    
    # Dropout with p=0.1, training=False
    dropout_out = torch.nn.functional.dropout(linear_out, 0.1, False, False)
    
    # Addition
    add_out = dropout_out + addend
    
    # Return the result as expected by transformer pattern
    return (add_out,)

def replacement_args(bias, weight, x, addend):
    return (bias, weight, x, addend)

@triton.jit
def optimized_dropout_add_kernel(
    linear_out_ptr,
    addend_ptr,
    out_ptr,
    n_elements,
    p_dropout,
    block_size: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < n_elements
    
    # Load linear output and addend
    linear_out = tl.load(linear_out_ptr + offsets, mask=mask, other=0.0)
    addend = tl.load(addend_ptr + offsets, mask=mask, other=0.0)
    
    # Fast dropout implementation for training=False (fixed random pattern)
    # Generate random numbers using offset-based deterministic pattern
    # This gives us reproducible results and avoids random number generation overhead
    r = tl.rand(offsets)  # Returns float in [0, 1)
    
    # Apply dropout: keep probability = 1 - p_dropout
    scale = 1.0 / (1.0 - p_dropout)  # Scale to maintain expected value
    mask_keep = r > p_dropout
    dropout_out = linear_out * mask_keep * scale
    
    # Fused addition operation
    out_result = dropout_out + addend
    
    # Store result
    tl.store(out_ptr + offsets, out_result, mask=mask)

@torch.fx.wrap  
def kernel_wrapper(bias, weight, x, addend):
    # Use PyTorch's optimized linear operation
    linear_out = torch.nn.functional.linear(x, weight, bias)
    
    # Determine tensor shapes and compute total elements
    if linear_out.dim() == 2:
        # Shape [batch, features]
        batch_size, features = linear_out.shape
        n_elements = batch_size * features
    else:
        # Handle 3D tensors by flattening spatial dimensions
        batch_size = linear_out.shape[0]
        spatial_size = linear_out.shape[1] * linear_out.shape[2] if linear_out.dim() == 3 else 1
        n_elements = batch_size * spatial_size
    
    # Dropout probability
    p_dropout = 0.1
    
    # Determine optimal block size and grid size
    BLOCK_SIZE = 1024  # Can be autotuned
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(linear_out)
    
    # Launch optimized kernel that fuses dropout + addition
    optimized_dropout_add_kernel[(num_programs,)](
        linear_out_ptr=linear_out,
        addend_ptr=addend,
        out_ptr=out,
        n_elements=n_elements,
        p_dropout=p_dropout,
        block_size=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return kernel_wrapper