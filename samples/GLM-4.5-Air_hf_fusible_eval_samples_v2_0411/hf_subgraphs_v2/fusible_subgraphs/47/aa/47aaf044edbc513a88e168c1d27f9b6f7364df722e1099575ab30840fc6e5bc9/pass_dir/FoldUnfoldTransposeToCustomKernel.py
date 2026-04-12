import torch
import triton
import triton.language as tl

def pattern(x):
    # tmp_0 = in_0.contiguous()
    # tmp_1 = tmp_0.unsqueeze(-1)
    tmp_1 = x.unsqueeze(-1)
    tmp_2 = torch.nn.functional.unfold(tmp_1, kernel_size=[9, 1], dilation=1, padding=[4, 0], stride=1)
    # tmp_3 = tmp_2.transpose(1, 2)
    tmp_3 = tmp_2.transpose(1, 2)
    return tmp_3

def replacement_args(x):
    return (x,)

@triton.jit
def custom_unfold_transpose_kernel(
    x_ptr,
    out_ptr,
    batch,
    channels,
    length,
    output_channels,
    output_length,
    BLOCK_SIZE: tl.constexpr,
):
    """Custom Triton kernel for fused unsqueeze + unfold + transpose operations"""
    pid = tl.program_id(0)
    
    # Each program handles one output position in the sliding window
    output_idx = pid
    
    if output_idx >= output_length * batch:
        return
    
    # Calculate batch and position index
    batch_idx = output_idx // output_length
    pos_idx = output_idx % output_length
    
    # Base pointer for current batch (3D input: [batch, channels, length])
    batch_x_ptr = x_ptr + batch_idx * channels * length
    
    # Process each window position (9 elements)
    for k in range(9):
        # Calculate input position with padding
        input_pos = pos_idx + k - 4  # 4 is padding
        
        if 0 <= input_pos < length:
            # Load input element (effectively doing unsqueeze(-1) virtually)
            for c in range(0, channels, BLOCK_SIZE):
                # Calculate mask for this channel block
                mask = c + tl.arange(0, BLOCK_SIZE) < channels
                # Load input element (virtual last dimension of size 1)
                offset = input_pos + c * length
                x_value = tl.load(batch_x_ptr + offset, mask=mask, other=0.0)
                # Store to output (transposed layout): [batch, output_length, output_channels]
                # output_channels = channels * 9
                local_output_offset = (pos_idx * output_channels) + (c * 9) + k
                global_output_offset = batch_idx * (output_length * output_channels) + local_output_offset
                tl.store(out_ptr + global_output_offset, x_value, mask=mask)

@torch.fx.wrap
def custom_unfold_transpose(x):
    """Wrapper function for custom unfold + transpose kernel"""
    batch, channels, length = x.shape
    # Output dimensions: [batch, length-2, channels*9]
    output_length = length - 2  # Due to padding and kernel size
    output_channels = channels * 9
    
    # Determine optimal block size
    BLOCK_SIZE = min(32, max(8, channels // 16))
    
    # Calculate grid size
    total_elements = batch * output_length
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty((batch, output_length, output_channels), 
                     dtype=x.dtype, device=x.device)
    
    # Launch kernel
    custom_unfold_transpose_kernel[grid_size](
        x_ptr=x,
        out_ptr=out,
        batch=batch,
        channels=channels,
        length=length,
        output_channels=output_channels,
        output_length=output_length,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return custom_unfold_transpose