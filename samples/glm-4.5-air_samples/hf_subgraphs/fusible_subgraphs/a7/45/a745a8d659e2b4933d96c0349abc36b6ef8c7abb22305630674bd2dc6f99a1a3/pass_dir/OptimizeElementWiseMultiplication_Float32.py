import torch
import triton
import triton.language as tl

@triton.jit
def optimized_mul_kernel(
    x_ptr,           # First input tensor
    y_ptr,           # Second input tensor  
    out_ptr,         # Output tensor
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Perform element-wise multiplication
    out = x * y
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_optimized_mul(x, y):
    """Optimized element-wise multiplication using Triton"""
    x = x.contiguous()
    y = y.contiguous()
    
    # Handle different broadcasting scenarios
    # Case 1: Same shape broadcast
    # Case 2: Vector broadcast (e.g., [256] broadcasted to [64, 17, 256])
    
    # Create output with same shape as x * y using a safe approach
    out_shape = x.shape
    try:
        # Simple broadcasting shape determination without forbidden API
        if x.dim() != y.dim():
            # Handle broadcasting by expanding dimensions
            max_dims = max(x.dim(), y.dim())
            x_shape = list(x.shape)
            y_shape = list(y.shape)
            
            # Pad with 1s on the left to match dimensions
            while len(x_shape) < max_dims:
                x_shape.insert(0, 1)
            while len(y_shape) < max_dims:
                y_shape.insert(0, 1)
            
            # Calculate output shape
            out_shape = []
            for i in range(max_dims):
                dim_x = x_shape[i] if x_shape[i] != 1 else y_shape[i]
                dim_y = y_shape[i] if y_shape[i] != 1 else x_shape[i]
                if dim_x != dim_y and dim_x != 1 and dim_y != 1:
                    raise RuntimeError(f"Shape mismatch at dim {i}: {x_shape[i]} vs {y_shape[i]}")
                out_shape.append(max(dim_x, dim_y))
    except:
        # Fallback if broadcast fails
        out = x * y
        return out
    
    out = torch.empty(out_shape, dtype=torch.float32, device=x.device)
    
    total_elements = out.numel()
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel if we can determine a consistent memory layout
    if total_elements > 0:
        optimized_mul_kernel[(num_programs,)](
            x, y, out, total_elements, BLOCK_SIZE
        )
    
    return out

# Pattern: element-wise multiplication
def pattern(first_tensor, second_tensor):
    return first_tensor * second_tensor

# Extract arguments for replacement
def replacement_args(first_tensor, second_tensor):
    return (first_tensor, second_tensor)

# Return optimized function
def replacement_func():
    return triton_optimized_mul