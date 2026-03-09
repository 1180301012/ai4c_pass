import torch
import triton
import triton.language as tl

def pattern(bias, weight, x, addend):
    # Linear transformation: y = x @ weight.t() + bias
    linear_out = torch.nn.functional.linear(x, weight, bias)
    
    # Dropout with p=0.0 is effectively identity
    dropout_out = torch.nn.functional.dropout(linear_out, p=0.0, training=False)
    
    # Addition
    add_out = addend + dropout_out
    
    # Return both values as expected by the LINKX pattern
    return add_out, dropout_out

def replacement_args(bias, weight, x, addend):
    return (bias, weight, x, addend)

@triton.jit
def optimized_linear_dropout_add_kernel(
    linear_out_ptr,  # Pre-computed linear output (we rely on PyTorch's optimized linear)
    addend_ptr,
    out_ptr,
    dropout_out_ptr, 
    n_elements,
    block_size: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < n_elements
    
    # Load linear output (already computed by PyTorch's optimized linear)
    linear_out = tl.load(linear_out_ptr + offsets, mask=mask, other=0.0)
    
    # Load addend 
    addend = tl.load(addend_ptr + offsets, mask=mask, other=0.0)
    
    # Since dropout p=0.0 is identity, we skip it entirely and fuse linear + add
    dropout_result = linear_out
    
    # Addition operation (fused with linear)
    out_result = addend + linear_out
    
    # Store results
    tl.store(out_ptr + offsets, out_result, mask=mask)
    tl.store(dropout_out_ptr + offsets, dropout_result, mask=mask)

@torch.fx.wrap
def kernel_wrapper(bias, weight, x, addend):
    # Use PyTorch's optimized linear operation first
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
    
    # Determine optimal block size
    BLOCK_SIZE = 1024  # Can be autotuned
    
    # Calculate grid size
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensors with same shape as linear output
    out = torch.empty_like(linear_out)
    dropout_out = torch.empty_like(linear_out)
    
    # Launch optimized kernel that fuses dropout (identity) + addition
    optimized_linear_dropout_add_kernel[(num_programs,)](
        linear_out_ptr=linear_out,
        addend_ptr=addend,
        out_ptr=out,
        dropout_out_ptr=dropout_out,
        n_elements=n_elements,
        block_size=BLOCK_SIZE,
    )
    
    return out, dropout_out

def replacement_func():
    return kernel_wrapper