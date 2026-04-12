import torch
import triton
import triton.language as tl

def pattern(a, b, c):
    """Pattern: element-wise multiply followed by add for tensors"""
    result = a * b + c
    return result

def replacement_args(a, b, c):
    return (a, b, c)

@triton.jit
def fused_mul_add_kernel(
    a_ptr, b_ptr, c_ptr, out_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """Fused kernel for element-wise multiply followed by add"""
    # Each program handles a contiguous block
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensors with bounds checking  
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    
    # Fused computation: multiply then add
    result = a * b + c
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_mul_add(a, b, c):
    """Function that performs fused multiply-add using Triton kernel"""
    # Convert inputs to tensors if they aren't already (handles scalars)
    # Use Python type checking instead of torch.is_tensor
    a_tensor = torch.as_tensor(a) if not isinstance(a, torch.Tensor) else a
    b_tensor = torch.as_tensor(b) if not isinstance(b, torch.Tensor) else b  
    c_tensor = torch.as_tensor(c) if not isinstance(c, torch.Tensor) else c
    
    # Get the device from the first actual tensor
    device = None
    if isinstance(a_tensor, torch.Tensor) and a_tensor.is_cuda:
        device = a_tensor.device
    elif isinstance(b_tensor, torch.Tensor) and b_tensor.is_cuda:
        device = b_tensor.device
    elif isinstance(c_tensor, torch.Tensor) and c_tensor.is_cuda:
        device = c_tensor.device
    
    # Move scalar tensors to GPU if we have a CUDA device reference
    if device is not None:
        if not isinstance(a_tensor, torch.Tensor) or not a_tensor.is_cuda:
            a_tensor = torch.as_tensor(a_tensor, device=device)
        if not isinstance(b_tensor, torch.Tensor) or not b_tensor.is_cuda:
            b_tensor = torch.as_tensor(b_tensor, device=device)  
        if not isinstance(c_tensor, torch.Tensor) or not c_tensor.is_cuda:
            c_tensor = torch.as_tensor(c_tensor, device=device)
    
    # Get the shape from the first tensor
    if isinstance(a_tensor, torch.Tensor):
        reference_tensor = a_tensor
    elif isinstance(b_tensor, torch.Tensor):
        reference_tensor = b_tensor
    else:
        reference_tensor = c_tensor
    
    # Ensure all tensors are contiguous for efficient access
    a_tensor_contiguous = a_tensor.contiguous() if isinstance(a_tensor, torch.Tensor) else a_tensor
    b_tensor_contiguous = b_tensor.contiguous() if isinstance(b_tensor, torch.Tensor) else b_tensor
    c_tensor_contiguous = c_tensor.contiguous() if isinstance(c_tensor, torch.Tensor) else c_tensor
    
    # Create output with same shape and dtype as reference tensor
    output = torch.empty_like(reference_tensor)
    
    # Flatten for computation while preserving total elements
    n_elements = reference_tensor.numel()
    
    # Launch kernel
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_mul_add_kernel[(num_programs,)](
        a_tensor_contiguous, b_tensor_contiguous, c_tensor_contiguous, output,
        n_elements,
        BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return fused_mul_add