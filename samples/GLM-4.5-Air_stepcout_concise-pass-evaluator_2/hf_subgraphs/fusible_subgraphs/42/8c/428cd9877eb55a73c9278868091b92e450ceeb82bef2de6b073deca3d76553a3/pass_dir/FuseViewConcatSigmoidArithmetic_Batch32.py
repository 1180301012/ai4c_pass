import torch
import triton
import triton.language as tl

def pattern(conv_out, in_3, in_4):
    # Pattern for batch size 32: view(32, 1, -1) + concat + sigmoid + arithmetic
    reshaped = conv_out.view(32, 1, -1)
    concatenated = torch.cat([in_3, in_4, reshaped], 2)
    activated = concatenated.sigmoid()
    result = (activated - 0.25) * 3.141592653589793
    return result

def replacement_args(conv_out, in_3, in_4):
    return (conv_out, in_3, in_4)

@triton.jit
def fused_kernel(
    conv_out_ptr, in_3_ptr, in_4_ptr, out_ptr,
    in_3_size, in_4_size, total_size,
    alpha,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Each program handles a block of the concatenated output
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_size
    
    # Determine which tensor region each offset belongs to based on actual sizes
    in_3_region = offsets < in_3_size
    in_4_region = (offsets >= in_3_size) & (offsets < in_3_size + in_4_size)
    conv_region = offsets >= in_3_size + in_4_size
    
    # Load data from appropriate tensor region with proper masking
    in_3_data = tl.load(in_3_ptr + offsets, mask=(mask & in_3_region), other=0.0)
    in_4_data = tl.load(in_4_ptr + offsets - in_3_size, mask=(mask & in_4_region), other=0.0)
    conv_data = tl.load(conv_out_ptr + offsets - in_3_size - in_4_size, mask=(mask & conv_region), other=0.0)
    
    # Combine data from different tensor regions
    data = tl.where(in_3_region, in_3_data, tl.where(in_4_region, in_4_data, conv_data))
    
    # Apply fused operations: sigmoid + (x - 0.25) * alpha
    sigmoid = 1.0 / (1.0 + tl.exp(-data))
    result = (sigmoid - 0.25) * alpha
    
    # Store the result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_operation(conv_out, in_3, in_4):
    # Apply the view operation first
    conv_viewed = conv_out.view(conv_out.shape[0], 1, -1)
    
    # Calculate actual tensor sizes dynamically
    in_3_elements = in_3.numel()
    in_4_elements = in_4.numel()
    conv_elements = conv_viewed.numel()
    total_elements = in_3_elements + in_4_elements + conv_elements
    
    # Create output tensor
    out = torch.empty(total_elements, dtype=torch.float32, device=conv_out.device)
    
    # Triton kernel launch configuration
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the kernel with actual tensor sizes
    fused_kernel[(num_programs,)](
        conv_out_ptr=conv_viewed,
        in_3_ptr=in_3,
        in_4_ptr=in_4,
        out_ptr=out,
        in_3_size=in_3_elements,
        in_4_size=in_4_elements,
        total_size=total_elements,
        alpha=3.141592653589793,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape to match the expected output shape
    return out.view(conv_out.shape[0], 1, -1)

def replacement_func():
    return fused_operation