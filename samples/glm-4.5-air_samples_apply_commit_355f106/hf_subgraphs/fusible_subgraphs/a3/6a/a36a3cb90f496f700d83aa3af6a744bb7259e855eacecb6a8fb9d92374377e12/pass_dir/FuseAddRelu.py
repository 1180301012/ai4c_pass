import torch
import triton
import triton.language as tl

# Pattern matching function - matches addition followed by ReLU
def pattern(x, y):
    tmp_5 = x + y
    tmp_6 = torch.nn.functional.relu(tmp_5, inplace=False)
    return tmp_6

# Argument extraction function
def replacement_args(x, y):
    return (x, y)

# Optimized kernel that fuses addition and ReLU with autotune support
@triton.jit
def add_relu_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs with mask
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # fused computation: addition + ReLU  
    add_out = x + y
    relu_out = tl.max(add_out, 0.0)
    
    # Store result
    tl.store(out_ptr + offsets, relu_out, mask=mask)

@torch.fx.wrap
def fused_add_relu(x, y):
    # Handle different input shapes from both graphs
    if x.dim() == 4:
        # Reshape to 2D for vectorized processing: [N*C, H*W]
        N, C, H, W = x.shape
        x_reshaped = x.reshape(N * C, H * W)
        y_reshaped = y.reshape(N * C, H * W)
        
        out_reshaped = torch.empty_like(x_reshaped)
        
        # Calculate optimal block size
        total_elements = x_reshaped.numel()
        BLOCK_SIZE = 1024  # Good balance for GPU occupancy
        num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        # Launch kernel
        add_relu_kernel[(num_programs,)](
            x_ptr=x_reshaped,
            y_ptr=y_reshaped,
            out_ptr=out_reshaped,
            n_elements=total_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        # Reshape back to original dimensions
        return out_reshaped.reshape(N, C, H, W)
    else:
        # Fallback for 1D case
        out = torch.empty_like(x)
        total_elements = x.numel()
        BLOCK_SIZE = 1024
        num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        add_relu_kernel[(num_programs,)](
            x_ptr=x,
            y_ptr=y,
            out_ptr=out,
            n_elements=total_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return out

# Replacement function
def replacement_func():
    return fused_add_relu