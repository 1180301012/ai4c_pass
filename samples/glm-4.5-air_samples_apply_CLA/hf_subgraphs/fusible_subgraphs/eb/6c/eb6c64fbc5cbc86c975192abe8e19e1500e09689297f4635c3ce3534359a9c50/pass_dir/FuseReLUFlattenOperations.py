import torch
import triton
import triton.language as tl

def pattern(x):
    # Match ReLU followed by flatten operations
    tmp_0 = torch.nn.functional.relu(x, inplace=True)
    tmp_1 = tmp_0.flatten(1, -1)
    return tmp_1

def replacement_args(x):
    return (x,)

@triton.jit
def relu_flatten_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused ReLU + Flatten kernel"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU operation
    out = tl.maximum(x, 0.0)
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_relu_flatten(x):
    """Fused ReLU + Flatten operation"""
    # Get input tensor properties
    original_shape = x.shape
    
    # Flatten dimensions 1 to -1 to get 1D tensor for kernel processing
    # This creates the correct output shape for flatten(1, -1)
    x_flat = x.flatten(1)  # This gives us the flattened 1D tensor
    
    # Prepare output tensor (same shape as the intended flattened result)
    output_shape = [original_shape[0], -1]  # [batch_size, flattened_dims]
    
    # Create output tensor with correct shape
    total_elements = x_flat.numel()
    out = torch.empty(total_elements, dtype=x.dtype, device=x.device, requires_grad=x.requires_grad)
    
    # Check if we need to handle the case where the flattened dimension might be 0
    if total_elements == 0:
        # Handle edge case - return empty tensor with correct shape
        return x.view(*output_shape)
    
    # Triton kernel launch parameters
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the fused kernel
    relu_flatten_kernel[(num_programs,)](
        x_ptr=x_flat,
        out_ptr=out,
        n_elements=total_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape output to match target flatten(1, -1) shape
    return out.view(*output_shape)

def replacement_func():
    return fused_relu_flatten