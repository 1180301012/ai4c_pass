import torch
import triton
import triton.language as tl

# Pattern matching function - matches the computation exactly
def pattern(in_0, in_1):
    tmp_0 = torch.cat([in_0, in_1], dim=1)
    slice_spec = (slice(None, None, None), slice(None, 672, None), slice(None, None, None), slice(None, None, None))
    tmp_1 = tmp_0[slice_spec] 
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return tmp_1, tmp_2

# Argument extraction function - extracts inputs needed for replacement
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Simple optimized kernel that avoids concatenation + + slicing
@triton.jit
def simple_kernel(
    in0_ptr, in1_ptr,
    out1_ptr, out2_ptr,  
    n_batch, n_channels_in0, n_channels_in1, height, width,
    target_channels,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    
    # Calculate indices
    batch_id = pid
    h_offset = pid_h * BLOCK_SIZE  
    w_offset = pid_w * BLOCK_SIZE
    
    # Bounds checking
    if batch_id >= n_batch or h_offset >= height or w_offset >= width:
        return
        
    # Local coordinates
    h = h_offset + tl.arange(0, BLOCK_SIZE)
    w = w_offset + tl.arange(0, BLOCK_SIZE)
    h_grid = h[:, None]
    w_grid = w[None, :]
    
    # Masks
    mask_h = h_grid < height
    mask_w = w_grid < width
    mask = mask_h & mask_w
    
    # Process each target channel
    for c in range(target_channels):
        # Determine source
        if c < n_channels_in0:
            src_ptr = in0_ptr
            src_channel = c
        else:
            src_ptr = in1_ptr
            src_channel = c - n_channels_in0
            
        # Skip if source doesn't exist
        if c >= n_channels_in0 or src_channel >= n_channels_in1:
            continue
            
        # Calculate offsets
        src_offset = src_ptr + batch_id * n_channels_in0 * height * width + src_channel * height * width
        
        # Load data
        if tl.sum(mask.to(tl.int32)) > 0:
            vals = tl.load(src_offset + h_grid * width + w_grid, mask=mask, other=0.0)
            
            # Store sliced output
            out1_offset = out1_ptr + batch_id * target_channels * height * width + c * height * width
            tl.store(out1_offset + h_grid * width + w_grid, vals, mask=mask)
            
            # Accumulate for mean (using atomic add for safety)
            mean_offset = out2_ptr + batch_id * target_channels
            tl.atomic_add(mean_offset + c, tl.sum(vals))

@torch.fx.wrap  
def simple_fused_op(in_0, in_1):
    batch, channels_in0, height, width = in_0.shape
    channels_in1 = in_1.shape[1]
    target_channels = 672
    
    # Validate shapes
    assert in_0.shape[0] == in_1.shape[0]
    assert in_0.shape[2] == in_1.shape[2] 
    assert in_0.shape[3] == in_1.shape[3]
    
    # Create outputs
    sliced_output = torch.empty((batch, target_channels, height, width), dtype=in_0.dtype, device=in_0.device)
    mean_output = torch.zeros((batch, target_channels, 1, 1), dtype=torch.float32, device=in_0.device)
    
    # Launch kernel
    grid = (batch, triton.cdiv(height, 16), triton.cdiv(width, 16))
    simple_kernel[grid](
        in0_ptr=in_0,
        in1_ptr=in_1,
        out1_ptr=sliced_output,
        out2_ptr=mean_output,
        n_batch=batch,
        n_channels_in0=channels_in0,
        n_channels_in1=channels_in1, 
        height=height,
        width=width,
        target_channels=target_channels,
        BLOCK_SIZE=16
    )
    
    # Finalize mean computation
    final_mean = mean_output.reshape(batch, target_channels) / (height * width)
    mean_output = final_mean.reshape(batch, target_channels, 1, 1)
    
    return sliced_output, mean_output

# Replacement function - returns the optimized kernel function
def replacement_func():
    return simple_fused_op