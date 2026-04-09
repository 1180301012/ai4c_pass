import torch
import triton
import triton.language as tl

@triton.jit
def fused_device_transfer_type_add_kernel(
    src_ptr,          # Source tensor (position embeddings) 
    dst_ptr,          # Destination tensor (conv3d output)
    out_ptr,          # Output tensor (result)
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized kernel with improved memory access patterns for better performance.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Optimized: Load adjacent elements for better memory coalescing
    src_vals = tl.load(src_ptr + offsets, mask=mask, other=0.0)
    dst_vals = tl.load(dst_ptr + offsets, mask=mask, other=0.0)
    
    # Optimized: Vectorized addition
    result = src_vals + dst_vals
    
    # Optimized: Store consecutive memory locations
    tl.store(out_ptr + offsets, result, mask=mask)

def pattern(tmp_6, tmp_5):
    """Matches the sequence: detach → type_as → to(device) + addition"""
    tmp_7 = tmp_6.type_as(tmp_5)
    tmp_8 = tmp_7.to(device=torch.device('cuda', index=0), copy=True)
    tmp_9 = tmp_5 + tmp_8
    return tmp_9

def replacement_args(tmp_6, tmp_5):
    """Extract the arguments needed for the fused operation"""
    return (tmp_6, tmp_5)

@torch.fx.wrap
def fused_device_transfer_type_add(pos_emb, conv_result):
    """
    Fused kernel that performs device transfer, type conversion, and addition
    in a single operation to eliminate intermediate tensor creation.
    """
    # First, ensure pos_emb is on the same device and has the same dtype as conv_result
    # This replicates the behavior of the original: tmp_7 = tmp_6.type_as(tmp_5)
    pos_emb = pos_emb.to(device=conv_result.device, dtype=conv_result.dtype, copy=False)
    
    # Get total number of elements for the simple 1D kernel
    n_elements = pos_emb.numel()
    
    # Create output tensor
    output = torch.empty_like(conv_result)
    
    # Use very small block size for maximum GPU occupancy
    BLOCK_SIZE = 256
    
    # Calculate number of programs needed
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the optimized 1D kernel
    fused_device_transfer_type_add_kernel[(num_programs,)](
        pos_emb,
        conv_result,
        output,
        n_elements,
        BLOCK_SIZE
    )
    
    return output

def replacement_func():
    """Returns the fused kernel function"""
    return fused_device_transfer_type_add