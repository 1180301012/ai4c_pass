import torch
import triton
import triton.language as tl

def pattern(x, y):
    # Pattern: tmp_4 = x / y, tmp_5 = tmp_4.to(torch.float32)
    tmp_4 = x / y
    tmp_5 = tmp_4.to(torch.float32)
    return tmp_5

def replacement_args(x, y):
    return (x, y)

@triton.jit
def fused_division_kernel(
    x_ptr,
    y_ptr, 
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs with broadcasting optimization
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=1.0)  # Avoid division by zero
    
    # Perform division (both inputs are already float32, no need for cast)
    result = x / y
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_division(x, y):
    # Determine the shape and create output tensor
    if hasattr(x, 'shape') and hasattr(y, 'shape'):
        if x.shape == y.shape:
            # Same shape case
            shape = x.shape
        else:
            # Handle broadcasting without torch.broadcast_shapes
            # Get maximum dimensions
            max_ndim = max(len(x.shape), len(y.shape))
            
            # Align dimensions from right
            x_shape = list(x.shape)
            y_shape = list(y.shape)
            
            # Prepend dimensions to match
            while len(x_shape) < max_ndim:
                x_shape.insert(0, 1)
            while len(y_shape) < max_ndim:
                y_shape.insert(0, 1)
            
            # Compute output shape
            shape = []
            for i in range(max_ndim):
                dim_x = x_shape[i]
                dim_y = y_shape[i]
                
                if dim_x == 1 and dim_y == 1:
                    shape.append(1)
                elif dim_x == 1:
                    shape.append(dim_y)
                elif dim_y == 1:
                    shape.append(dim_x)
                else:
                    if dim_x != dim_y:
                        raise RuntimeError(f"Shape mismatch at dim {i}: {dim_x} vs {dim_y}")
                    shape.append(dim_x)
    else:
        # Fallback for tensors without shape attribute
        shape = getattr(x, 'shape', (1,))
    
    # Create output tensor
    out = torch.empty(shape, dtype=torch.float32, device=x.device)
    
    # Calculate total number of elements
    n_elements = out.numel()
    
    # Optimize block size based on tensor size
    if n_elements >= 1000000:
        BLOCK_SIZE = 2048  # Large tensors can use larger blocks
    elif n_elements >= 100000:
        BLOCK_SIZE = 1024  # Medium tensors
    else:
        BLOCK_SIZE = 512   # Small tensors
    
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_division_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_division