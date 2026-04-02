import torch
import triton
import triton.language as tl

# Pattern matching function for the entire computation sequence
def pattern(in_4, in_1, in_3, in_0, in_2):
    """
    Match the entire sequence:
    einsum = torch.einsum('bchj,bhwj->bchw', in_4, in_1)
    in_3 += einsum
    in_5 = in_3
    tmp_3 = in_5 * in_0
    tmp_4 = tmp_3 + in_2
    return tmp_4  # intermediate result, will be followed by .contiguous()
    """
    # Einsum contraction
    einsum = torch.functional.einsum('bchj,bhwj->bchw', in_4, in_1)
    
    # In-place addition  
    in_3 += einsum
    
    # Element-wise operations
    tmp_3 = in_3 * in_0  # in_5 is replaced by in_3 directly
    tmp_4 = tmp_3 + in_2
    
    return tmp_4

# Argument extraction function
def replacement_args(in_4, in_1, in_3, in_0, in_2):
    return (in_4, in_1, in_3, in_0, in_2)

@triton.jit
def fused_computation_kernel(
    in_4_ptr, in_1_ptr, in_3_ptr, in_0_value, in_2_ptr,
    out_ptr,
    batch_size, channels_h, height_out, width_out, height_in,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Total output elements
    total_output_elements = batch_size * channels_h * height_out * width_out
    if pid >= total_output_elements:
        return
    
    # Parse program ID for output coordinates
    out_idx = pid
    b = out_idx // (channels_h * height_out * width_out)
    out_idx = out_idx % (channels_h * height_out * width_out)
    c = out_idx // (height_out * width_out)
    out_idx = out_idx % (height_out * width_out)
    h = out_idx // width_out
    w = out_idx % width_out
    
    # Initialize accumulation tensor
    einsum_result = tl.zeros([], dtype=tl.float32)
    
    # Stage 1: Einsum contraction (bchj,bhwj->bchw)
    for j in range(height_in):
        # From in_4: [b, c, h_in, j] access
        in_4_offset = b * (channels_h * height_in * width_out) + c * (height_in * width_out) + j * width_out + w
        val_in_4 = tl.load(in_4_ptr + in_4_offset, mask=(j < height_in), other=0.0)
        
        # From in_1: [b, h, w, j] access  
        in_1_offset = b * (height_out * width_out * height_in) + h * (width_out * height_in) + w * height_in + j
        val_in_1 = tl.load(in_1_ptr + in_1_offset, mask=(j < height_in), other=0.0)
        
        einsum_result += val_in_4 * val_in_1
    
    # Stage 2: Add to in_3 and apply element-wise operations
    # Load in_3 element at position [b, c, h, w]
    in_3_offset = b * (channels_h * height_out * width_out) + c * (height_out * width_out) + h * width_out + w
    in_3_val = tl.load(in_3_ptr + in_3_offset)
    
    # Load in_2 element at position [b, c, h, w]
    in_2_offset = b * (channels_h * height_out * width_out) + c * (height_out * width_out) + h * width_out + w
    in_2_val = tl.load(in_2_ptr + in_2_offset)
    
    # Final computation: ((in_3 + einsum_result) * in_0_value) + in_2_val
    result = ((in_3_val + einsum_result) * in_0_value) + in_2_val
    
    # Store final result
    tl.store(out_ptr + out_idx, result)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap  
def fused_computation(in_4, in_1, in_3, in_0, in_2):
    batch_size, channels_h, height_in, width_out = in_4.shape
    _, height_out, width_out, height_in_1 = in_1.shape
    _, channels_in3, height_out_in3, width_out_in3 = in_3.shape
    
    # Ensure consistent dimensions
    height_out = height_out_1
    assert height_in == height_in_1, f"Height mismatch: {height_in} vs {height_in_1}"
    assert channels_h == channels_in3, f"Channels mismatch: {channels_h} vs {channels_in3}"
    assert height_out == height_out_in3, f"Height mismatch: {height_out} vs {height_out_in3}" 
    assert width_out == width_out_in3, f"Width mismatch: {width_out} vs {width_out_in3}"
    
    # Output shape matches in_3 shape: [batch_size, channels_h, height_out, width_out]
    out = torch.empty_like(in_3)
    
    BLOCK_SIZE = 1024
    total_elements = batch_size * channels_h * height_out * width_out
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_computation_kernel[(num_programs,)](
        in_4_ptr=in_4,
        in_1_ptr=in_1, 
        in_3_ptr=in_3,
        in_0_value=in_0.item() if in_0.numel() == 1 else in_0,
        in_2_ptr=in_2,
        out_ptr=out,
        batch_size=batch_size,
        channels_h=channels_h,
        height_out=height_out,
        width_out=width_out,
        height_in=height_in,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_computation