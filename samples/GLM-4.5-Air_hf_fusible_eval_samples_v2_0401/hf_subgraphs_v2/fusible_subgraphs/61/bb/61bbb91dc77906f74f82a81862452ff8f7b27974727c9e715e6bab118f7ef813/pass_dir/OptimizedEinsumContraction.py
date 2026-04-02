import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_4, in_1):
    """Match einsum contraction: bchj,bhwj->bchw"""
    return torch.functional.einsum('bchj,bhwj->bchw', in_4, in_1)

# Argument extraction function
def replacement_args(in_4, in_1):
    return (in_4, in_1)

@triton.heuristics({"BLOCK_SIZE": lambda kwargs: kwargs["batch_size"] * kwargs["channels_h"] * 32})
@triton.jit
def optimized_einsum_kernel(
    in_4_ptr, in_1_ptr, out_ptr,
    batch_size, channels_h, height_in, width_out, height_out,
    dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Total number of output elements = batch_size * channels_h * height_out * width_out
    total_output_elements = batch_size * channels_h * height_out * width_out
    if pid >= total_output_elements:
        return
    
    # Parse program ID to get output coordinates: [b, c, h_out, w_out]
    out_idx = pid
    b = out_idx // (channels_h * height_out * width_out)
    out_idx = out_idx % (channels_h * height_out * width_out)
    c = out_idx // (height_out * width_out)
    out_idx = out_idx % (height_out * width_out)
    h_out = out_idx // width_out
    w_out = out_idx % width_out
    
    # Initialize output accumulator with correct data type
    result = tl.zeros([], dtype=dtype)
    
    # Contract over dimension j
    for j in range(height_in):
        # Calculate input tensor addresses with correct strides
        # in_4 shape: [batch, channels_h, height_in, width_out] → b, c, j, w_out
        in_4_stride_batch = channels_h * height_in * width_out
        in_4_stride_channel = height_in * width_out
        in_4_stride_height = width_out
        in_4_offset = b * in_4_stride_batch + c * in_4_stride_channel + j * in_4_stride_height + w_out
        
        # in_1 shape: [batch, height_out, width_out, height_in] → b, h_out, w_out, j
        in_1_stride_batch = height_out * width_out * height_in
        in_1_stride_height = width_out * height_in
        in_1_stride_width = height_in
        in_1_offset = b * in_1_stride_batch + h_out * in_1_stride_height + w_out * in_1_stride_width + j
        
        # Load elements without bounds checking since we use fixed strides
        val_in_4 = tl.load(in_4_ptr + in_4_offset)
        val_in_1 = tl.load(in_1_ptr + in_1_offset)
        
        # Multiply and accumulate
        result += val_in_4 * val_in_1
    
    # Store result with correct output stride
    out_stride_batch = channels_h * height_out * width_out
    out_stride_channel = height_out * width_out
    out_stride_height = width_out
    out_offset = b * out_stride_batch + c * out_stride_channel + h_out * out_stride_height + w_out
    tl.store(out_ptr + out_offset, result)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def optimized_einsum(in_4, in_1):
    batch_size, channels_h, height_in, width_out = in_4.shape
    _, height_out, width_out_1, height_in_1 = in_1.shape
    
    # Validate dimensions
    assert width_out == width_out_1, f"Width mismatch: {width_out} vs {width_out_1}"
    assert height_in == height_in_1, f"Height mismatch: {height_in} vs {height_in_1}"
    
    # Output shape: [batch_size, channels_h, height_out, width_out]
    out = torch.empty((batch_size, channels_h, height_out, width_out), dtype=in_4.dtype, device=in_4.device)
    
    # Determine Triton data type
    if in_4.dtype == torch.bfloat16:
        dtype = tl.bfloat16
    elif in_4.dtype == torch.float16:
        dtype = tl.float16  
    else:
        dtype = tl.float32
    
    BLOCK_SIZE = 1024
    total_output_elements = batch_size * channels_h * height_out * width_out
    num_programs = (total_output_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_einsum_kernel[(num_programs,)](
        in_4_ptr=in_4,
        in_1_ptr=in_1,
        out_ptr=out,
        batch_size=batch_size,
        channels_h=channels_h,
        height_in=height_in,
        width_out=width_out,
        height_out=height_out,
        dtype=dtype,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return optimized_einsum