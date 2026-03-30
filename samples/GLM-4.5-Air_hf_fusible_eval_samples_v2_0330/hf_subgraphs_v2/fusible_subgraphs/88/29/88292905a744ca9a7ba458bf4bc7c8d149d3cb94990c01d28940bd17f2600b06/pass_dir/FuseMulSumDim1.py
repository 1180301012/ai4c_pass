import torch
import triton
import triton.language as tl

def pattern(multiplier, operand):
    # Original computation:
    # tmp_3 = multiplier  # shape [8, 2, 128, 1, 1]
    # tmp_4 = tmp_3 * operand  # operand has shape [8, 2, 128, 120, 160], broadcasting applies
    # tmp_5 = torch.sum(tmp_4, dim=1)
    
    # Fuse multiplication + sum into single operation
    mul_out = multiplier * operand
    sum_out = torch.sum(mul_out, dim=1)
    return sum_out

def replacement_args(multiplier, operand):
    return (multiplier, operand)

@triton.jit
def fused_mul_sum_kernel(
    multiplier_ptr,
    operand_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one channel (dim=1 after sum)
    channel_idx = tl.program_id(0)
    
    # Calculate output stride for this channel
    output_offset = channel_idx * height * width
    offsets = output_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < height * width
    
    # Load multiplier data for this channel and batch
    # multiplier has shape [batch, channels, spatial...], we assume spatial dimensions are [1, 1]
    multiplier_val = tl.load(multiplier_ptr + channel_idx, mask=channel_idx < batch_size * channels, other=0.0)
    
    # Load operand data for this channel across all spatial positions
    # operand has shape [batch, channels, height, width]
    operand_base = channel_idx * height * width
    operand_vals = tl.load(operand_ptr + operand_base + offsets, mask=mask, other=0.0)
    
    # Perform multiplication and sum across batch dimension
    # We need to sum over the batch dimension (dim=0)
    
    # For each spatial position, sum over batch and channel dimensions
    total_sum = 0.0
    for batch_idx in range(batch_size):
        for ch_idx in range(channels):
            # Calculate offset in multiplier tensor
            mult_offset = batch_idx * channels + ch_idx
            multiplier_val = tl.load(multiplier_ptr + mult_offset, mask=mult_offset < batch_size * channels, other=0.0)
            
            # Load corresponding operand data
            operand_offset = (batch_idx * channels + ch_idx) * height * width + offsets
            operand_vals = tl.load(operand_ptr + operand_offset, mask=mask, other=0.0)
            
            # Multiply and accumulate
            total_sum += multiplier_val * operand_vals
    
    # Store the result
    tl.store(out_ptr + offsets, total_sum, mask=mask)

@torch.fx.wrap
def fused_mul_sum(multiplier, operand):
    # Get tensor properties
    batch_size = operand.shape[0]
    channels = operand.shape[1]  # This is the dimension we're summing over
    height = operand.shape[2]
    width = operand.shape[3]
    
    # Multiplier shape should be compatible: [batch, channels, 1, 1] or broadcastable
    # We assume multiplier has shape that broadcasts to [batch, channels, height, width]
    
    # Output shape after sum along dim=1: [batch, height, width]
    # But the pattern shows we return sum_out which should be [8, 128, 120, 160]
    # Let's check the original computation: summing along dim=1 gives [batch, channels_except_1, spatial...]
    # In our case: [8, 128, 120, 160] where 128 comes from the third dimension of multiplier
    output_shape = [batch_size, channels, height, width]
    out = torch.empty(output_shape, dtype=operand.dtype, device=operand.device)
    
    # Flatten tensors for kernel
    multiplier_flat = multiplier.reshape(-1)
    operand_flat = operand.reshape(-1)
    out_flat = out.reshape(-1)
    
    # Determine kernel launch configuration  
    # We have one program per output element (after sum)
    total_output_elements = batch_size * height * width  # After summing over channels
    BLOCK_SIZE = 1024
    num_programs = (total_output_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_mul_sum_kernel[(num_programs,)](
        multiplier_ptr=multiplier_flat,
        operand_ptr=operand_flat,
        out_ptr=out_flat,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape output to match expected [batch, height, width] after sum along dim=1
    # But looking at the original pattern, the output should be [8, 128, 120, 160]
    # This suggests the sum operation reduces the channel dimension (dim=1)
    final_output_shape = [batch_size, height, width]
    final_output = out.view(final_output_shape)
    
    return final_output

def replacement_func():
    return fused_mul_sum