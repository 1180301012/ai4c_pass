import torch
import triton
import triton.language as tl

# Pattern matching function for view + expand operations
def pattern(x):
    """Matches the view + expand pattern: view to [1,2,1,8,8] then expand to [1,2,64,8,8]"""
    tmp_2 = x.view(1, 2, 1, 8, 8)
    tmp_3 = tmp_2.expand(1, 2, 64, 8, 8)
    return tmp_3

# Argument extraction function
def replacement_args(x):
    return (x,)

# Optimized kernel for view + expand (broadcasting) operations
@triton.jit
def broadcast_kernel(
    x_ptr,
    out_ptr,
    out_n0,
    out_n1,
    out_n2,
    out_n3,
    out_n4,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel for efficient broadcasting from [1,2,8,8] to [1,2,64,8,8]"""
    # Use 3D grid: batch, spatial_dim3, spatial_dim4
    pid0 = tl.program_id(0)  # batch dimension  
    pid3 = tl.program_id(1)  # spatial dim 3
    pid4 = tl.program_id(2)  # spatial dim 4
    
    # For each position, we need to broadcast along the middle dimension (0->64)
    for i in range(64):  # Handle the broadcast dimension
        # Load from original tensor [1,2,8,8]
        input_offset = pid0 * (2 * 8 * 8) + tl.program_id(1) * (8 * 8) + pid3 * 8 + pid4
        
        # Load value with mask
        input_val = tl.load(x_ptr + input_offset, mask=(tl.program_id(1) < 2) & (pid3 < 8) & (pid4 < 8), other=0.0)
        
        # Store to expanded tensor [1,2,64,8,8]  
        # Calculate broadcast offset: handle all 5 dimensions
        output_offset = (pid0 * (2 * 64 * 8 * 8) + 
                        tl.program_id(1) * (64 * 8 * 8) +  # channel dim
                        i * (8 * 8) +  # broadcast dim (0->63)
                        pid3 * 8 +     # spatial dim 3
                        pid4)          # spatial dim 4
        
        tl.store(out_ptr + output_offset, input_val)

@torch.fx.wrap
def efficient_broadcast(x):
    """Wrapper function for optimized broadcasting from view + expand"""
    n0, n1, n2, n3 = x.shape  # [1, 2, 8, 8]
    
    # Output shape after expand: [1, 2, 64, 8, 8]
    out_shape = [n0, n1, 64, n3, 8]
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Launch grid: [batch, channel, spatial_dim3] - 3D max for Triton
    # We'll handle the other dimensions in the kernel
    grid = lambda meta: (n0, n1, n3)
    
    broadcast_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        out_n0=n0,
        out_n1=n1,
        out_n2=64,
        out_n3=n3,
        out_n4=8,
        BLOCK_SIZE=1
    )
    
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return efficient_broadcast