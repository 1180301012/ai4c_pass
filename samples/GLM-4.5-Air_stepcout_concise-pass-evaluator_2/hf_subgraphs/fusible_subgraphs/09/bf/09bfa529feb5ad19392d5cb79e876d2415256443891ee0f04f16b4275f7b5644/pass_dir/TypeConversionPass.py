import torch
import triton
import triton.language as tl

# Pattern matching function for addition + type conversion (where dropout is no-op)
def pattern(in_0, in_1):
    """Match: addition -> type conversion (when dropout has no effect)"""
    tmp_0 = in_0 + in_1
    tmp_1 = torch.nn.functional.softmax(tmp_0, dim=-1)
    # This pattern matches when dropout doesn't change the tensor
    tmp_2 = tmp_1
    tmp_3 = tmp_2.to(torch.float32)
    return tmp_3

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized kernel for addition + type conversion with smart dtype handling
@triton.jit
def smart_type_conversion_kernel(
    x_ptr,
    y_ptr, 
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    """Optimized kernel: addition with smart type conversion"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensors
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Perform addition and handle type conversion efficiently
    added = x + y
    
    # If input is already float32, operation is minimal
    # Load as float32 to avoid unnecessary casts
    result = tl.cast(added, tl.float32)
    
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def smart_add_with_type_conversion(x, y):
    """Smart addition with optimal type conversion"""
    # Check if tensors are already compatible float32
    if x.dtype == torch.float32 and y.dtype == torch.float32:
        # Direct addition, both tensors already correct type
        out = torch.empty_like(x)
        n_elements = x.numel()
        
        if n_elements >= 1024:  # Use Triton for larger tensors
            BLOCK_SIZE = min(512, n_elements)
            num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
            
            smart_type_conversion_kernel[(num_programs,)](
                x_ptr=x,
                y_ptr=y,
                out_ptr=out,
                n_elements=n_elements,
                BLOCK_SIZE=BLOCK_SIZE
            )
            return out
        else:
            return x + y  # Small tensor, use direct PyTorch
    
    # Fallback for other cases - maintain compatibility
    tmp_0 = x + y
    tmp_1 = torch.nn.functional.softmax(tmp_0, dim=-1)
    tmp_3 = tmp_1.to(torch.float32)
    return tmp_3

# Replacement function
def replacement_func():
    # Return a closure for optimal type conversion
    def kernel_optimal(x, y):
        return smart_add_with_type_conversion(x, y)
    
    return kernel_optimal