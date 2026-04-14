import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_1):
    """
    Match the split + unsqueeze pattern:
    split = torch.split(in_1, [512, 512, 128], dim=2)
    tmp_5 = split[2]  
    tmp_6 = tmp_5.unsqueeze(2)
    """
    split = torch.split(in_1, [512, 512, 128], dim=2)
    tmp_5 = split[2]
    tmp_6 = tmp_5.unsqueeze(2)
    return tmp_6

# Argument extraction function  
def replacement_args(in_1):
    return (in_1,)

# Optimized kernel - fuse split and unsqueeze operations
@triton.jit
def split_unsqueeze_fused_kernel(
    in_ptr,
    out_ptr,
    batch_size,
    height,
    width_512,
    width_128,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one batch element
    batch_idx = tl.program_id(0)
    height_idx = tl.program_id(1)
    
    # Calculate the offset for this program
    in_offset = (batch_idx * height + height_idx) * (width_512 + width_512 + width_128)
    out_offset = (batch_idx * height + height_idx) * width_128
    
    # Load the 128-width portion directly with unsqueeze effect
    offset = in_offset + width_512 + width_512 + tl.arange(0, BLOCK_SIZE)
    mask = offset < (in_offset + width_512 + width_512 + width_128)
    
    data = tl.load(in_ptr + offset, mask=mask, other=0.0)
    
    # Store the result (effectively unsqueezed to 4D)  
    tl.store(out_ptr + out_offset, data, mask=tl.arange(0, BLOCK_SIZE) < width_128)

# Kernel wrapper
@torch.fx.wrap
def split_unsqueeze_fused(in_1):
    batch_size, height, total_width = in_1.shape
    width_512 = 512
    width_128 = 128
    
    # Validate input dimensions
    if total_width != width_512 + width_512 + width_128:
        raise ValueError(f"Input width must be {width_512 + width_512 + width_128}, got {total_width}")
    
    # Calculate optimal block size
    BLOCK_SIZE = min(1024, width_128)
    num_batches = (batch_size + 1)  # Process each batch    
    num_height = (height + 1)      # Process each height
    
    out = torch.empty((batch_size, height, 1, width_128), dtype=in_1.dtype, device=in_1.device)
    
    # Launch kernel
    split_unsqueeze_fused_kernel[(num_batches, num_height)](
        in_ptr=in_1,
        out_ptr=out,  
        batch_size=batch_size,
        height=height,
        width_512=width_512,
        width_128=width_128,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

# Replacement function
def replacement_func():
    return split_unsqueeze_fused