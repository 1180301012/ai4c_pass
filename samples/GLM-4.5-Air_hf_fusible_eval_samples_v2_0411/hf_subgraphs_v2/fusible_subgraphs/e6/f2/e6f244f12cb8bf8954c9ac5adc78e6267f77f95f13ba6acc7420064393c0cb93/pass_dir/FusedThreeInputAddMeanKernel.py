import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    tmp_0 = in_1 + in_2
    tmp_0 = tmp_0 + in_0
    tmp_1 = tmp_0
    tmp_2 = tmp_1.mean((2, 3), keepdim = True)
    return (tmp_1, tmp_2)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def fused_three_input_add_mean_kernel(
    in0_ptr,
    in1_ptr,
    in2_ptr,
    out_ptr,
    mean_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Each program handles one complete sample (batch, channels)
    sample_idx = pid // channels
    channel_idx = pid % channels
    
    if sample_idx >= batch_size:
        return
    
    # Load all three input tensors for this sample/channel
    base_offset = sample_idx * channels * height * width + channel_idx * height * width
    in0_offset = base_offset + tl.arange(0, BLOCK_SIZE)
    in1_offset = base_offset + tl.arange(0, BLOCK_SIZE)
    in2_offset = base_offset + tl.arange(0, BLOCK_SIZE)
    
    # Ensure we don't read out of bounds
    mask = in0_offset < base_offset + height * width
    in0_data = tl.load(in0_ptr + in0_offset, mask=mask, other=0.0)
    in1_data = tl.load(in1_ptr + in1_offset, mask=mask, other=0.0)
    in2_data = tl.load(in2_ptr + in2_offset, mask=mask, other=0.0)
    
    # Add all three tensors and compute spatial sum/mean
    sum_data = in0_data + in1_data + in2_data
    spatial_sum = tl.sum(sum_data)
    spatial_count = tl.sum(tl.where(mask, 1, 0))
    
    # Store outputs
    # Store sum result
    tl.store(out_ptr + in0_offset, sum_data, mask=mask)
    
    # Store mean
    mean_offset = sample_idx * channels + channel_idx
    tl.store(mean_ptr + mean_offset, spatial_sum / spatial_count)

@torch.fx.wrap
def fused_three_input_add_mean(in_0, in_1, in_2):
    batch_size, channels, height, width = in_0.shape
    
    # Output for sum
    out_sum = torch.empty_like(in_0)
    
    # Output for mean
    out_mean = torch.empty((batch_size, channels, 1, 1), dtype=in_0.dtype, device=in_0.device)
    
    # Flatten to 2D for processing: [batch * channels, height * width]
    in0_flat = in_0.reshape(batch_size * channels, height * width)
    in1_flat = in_1.reshape(batch_size * channels, height * width)
    in2_flat = in_2.reshape(batch_size * channels, height * width)
    
    # Number of programs needed
    total_elements = batch_size * channels
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_three_input_add_mean_kernel[(num_programs,)](
        in0_ptr=in0_flat,
        in1_ptr=in1_flat,
        in2_ptr=in2_flat,
        out_ptr=out_sum.reshape(batch_size * channels, height * width),
        mean_ptr=out_mean.reshape(batch_size * channels),
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out_sum, out_mean

def replacement_func():
    return fused_three_input_add_mean