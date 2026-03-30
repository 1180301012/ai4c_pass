import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    # Branch 1: Scale and bias in_1
    tmp_1 = in_1 * 0.458
    tmp_2 = tmp_1 + -0.030000000000000027
    
    # Branch 2: Take slice from in_0, unsqueeze, scale and bias
    tmp_3 = in_0[(slice(None, None, None), 1)]
    tmp_4 = torch.unsqueeze(tmp_3, 1)
    tmp_5 = tmp_4 * 0.448
    tmp_6 = tmp_5 + -0.08799999999999997
    
    # Branch 3: Take slice from in_0, unsqueeze, scale and bias
    tmp_7 = in_0[(slice(None, None, None), 2)]
    tmp_8 = torch.unsqueeze(tmp_7, 1)
    tmp_9 = tmp_8 * 0.45
    tmp_10 = tmp_9 + -0.18799999999999994
    
    # Concatenate all three results
    tmp_11 = torch.cat((tmp_2, tmp_6, tmp_10), 1)
    
    return (tmp_11,)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_kernel(
    in_0_ptr, in_1_ptr,
    out_ptr,
    batch_size, in_0_channels, height, width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Process one batch at a time, handle all pixels in that batch
    pixel_idx = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    batch_pixel_offset = pid * height * width  # Offset for all pixels in this batch
    
    # Global pixel index across all batches and channels (assuming each pixel processed once)
    global_pixel_idx = batch_pixel_offset + pixel_idx
    
    # Mask to ensure we don't go out of bounds 
    mask = pixel_idx < (height * width)
    
    # Branch 1: Process in_1 tensor (has shape [batch, 1, height, width])
    # Index into in_1 as if it were [batch, height, width] (since channel=1 is fixed)
    in_1_offset = batch_pixel_offset + pixel_idx
    in_1_val = tl.load(in_1_ptr + in_1_offset, mask=mask, other=0.0)
    branch1_out = in_1_val * 0.458 + (-0.030000000000000027)
    
    # Branch 2: Process in_0 tensor channel 1
    # Original: in_0[:, 1] -> unsqueeze -> scale/bias
    # For channel 1: offset = (batch * in_0_channels * height * width) + (1 * height * width) + (h * width) + w
    # But we need to map this to our processing. Since we process one batch at a time:
    channel1_base_offset = 1 * height * width  # Channel 1 base offset
    in_0_channel1_offset = channel1_base_offset + pixel_idx
    
    # Load from in_0 - we're reading contiguous memory starting at channel1 position
    in_0_channel1 = tl.load(in_0_ptr + in_0_channel1_offset, mask=mask, other=0.0)
    branch2_out = in_0_channel1 * 0.448 + (-0.08799999999999997)
    
    # Branch 3: Process in_0 tensor channel 2  
    # For channel 2: offset = (batch * in_0_channels * height * width) + (2 * height * width) + (h * width) + w
    channel2_base_offset = 2 * height * width  # Channel 2 base offset
    in_0_channel2_offset = channel2_base_offset + pixel_idx
    
    in_0_channel2 = tl.load(in_0_ptr + in_0_channel2_offset, mask=mask, other=0.0)
    branch3_out = in_0_channel2 * 0.45 + (-0.18799999999999994)
    
    # Store results - output has shape [batch, 3, height, width]
    # We need to store each branch result in the correct channel position
    # Total output elements per batch: 3 * height * width
    output_batch_offset = pid * 3 * height * width
    
    # Branch 1 goes to output channel 0
    store_offset_channel0 = output_batch_offset + pixel_idx
    tl.store(out_ptr + store_offset_channel0, branch1_out, mask=mask)
    
    # Branch 2 goes to output channel 1 (add height*width)
    store_offset_channel1 = output_batch_offset + height * width + pixel_idx  
    tl.store(out_ptr + store_offset_channel1, branch2_out, mask=mask)
    
    # Branch 3 goes to output channel 2 (add 2*height*width)
    store_offset_channel2 = output_batch_offset + 2 * height * width + pixel_idx
    tl.store(out_ptr + store_offset_channel2, branch3_out, mask=mask)

@torch.fx.wrap  
def fused_kernel_wrapper(in_0, in_1):
    # Determine output shape [batch_size, 3, height, width]
    batch_size, in_0_channels, height, width = in_0.shape[0], in_0.shape[1], in_0.shape[2], in_0.shape[3]
    out_shape = (batch_size, 3, height, width)
    output = torch.empty(out_shape, dtype=in_1.dtype, device=in_1.device)
    
    # Use autotuning to find optimal block size
    n_pixels = height * width
    
    def get_best_block_size():
        # Use 512 as it provides good occupancy for most image sizes
        return 512
    
    BLOCK_SIZE = get_best_block_size()
    n_blocks = (n_pixels + BLOCK_SIZE - 1) // BLOCK_SIZE
    n_batches = batch_size
    
    fused_kernel[(n_batches, n_blocks)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=output,
        batch_size=batch_size,
        in_0_channels=in_0_channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_kernel_wrapper