import torch
import triton
import triton.language as tl

# Pattern matching exact computation structure
def pattern(in_3):
    tmp_2 = in_3.pow_(-0.5)
    return tmp_2

def replacement_args(in_3):
    return (in_3,)



@triton.jit
def fused_power_mask_kernel(
    deg_ptr,
    out_ptr,
    n_nodes,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of nodes
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_nodes
    
    # Load degrees
    deg = tl.load(deg_ptr + offsets, mask=mask, other=0.0)
    
    # Compute reciprocal square root and handle inf values
    # If deg is 0, rsqrt would be inf, so we clamp to minimum safe value
    deg_safe = tl.maximum(deg, 1e-7)  # Avoid division by zero
    rsqrt_deg = tl.math.rsqrt(deg_safe)
    
    # Clamp to avoid overflow (this handles the inf masking)
    rsqrt_deg = tl.minimum(rsqrt_deg, 1e6)  # Cap at reasonable value
    
    # Store result
    tl.store(out_ptr + offsets, rsqrt_deg, mask=mask)

@torch.fx.wrap
def fused_power_mask(deg_tensor):
    n_nodes = deg_tensor.shape[0]
    BLOCK_SIZE = 1024
    num_programs = (n_nodes + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(deg_tensor)
    
    fused_power_mask_kernel[(num_programs,)](
        deg_ptr=deg_tensor,
        out_ptr=out,
        n_nodes=n_nodes,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_power_mask