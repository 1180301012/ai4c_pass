import torch
import triton
import triton.language as tl

def pattern(x, y):
    # Match addition pattern that can be optimized with Triton
    result = x + y
    return result

def replacement_args(x, y):
    return (x, y)

@triton.jit
def triton_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs with proper masking
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Perform addition
    out = x + y
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_add_fused(x, y):
    # Fused addition using Triton for better performance
    # Handle both tensor-tensor and tensor-scalar operations
    # Only use Triton for large tensors to avoid kernel overhead
    
    # Handle scalar case
    if isinstance(y, (int, float)) or (isinstance(y, torch.Tensor) and y.numel() == 1):
        # If y is scalar, just add it directly to tensor x
        return x + y
    
    # For tensor-tensor operations, decide whether to use Triton or PyTorch
    if x.shape != y.shape:
        if x.numel() == 1:
            x = x.expand_as(y)
        elif y.numel() == 1:
            y = y.expand_as(x)
        else:
            raise ValueError("Shapes must be broadcastable")
    
    # Determine if we should use Triton based on tensor size
    # Use Triton only for tensors larger than 8192 elements to avoid kernel overhead
    if x.numel() < 8192:
        # For small tensors, use regular PyTorch addition (faster)
        return x + y
    
    # For large tensors, use Triton kernel
    x = x.contiguous()
    y = y.contiguous()
    
    # Launch Triton kernel
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    triton_add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return triton_add_fused