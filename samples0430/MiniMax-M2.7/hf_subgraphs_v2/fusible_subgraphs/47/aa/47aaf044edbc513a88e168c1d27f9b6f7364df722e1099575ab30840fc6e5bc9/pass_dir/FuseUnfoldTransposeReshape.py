import torch
import triton
import triton.language as tl

# Pattern matching: contiguous + unsqueeze + unfold + transpose + reshape + reshape
def pattern(in_0):
    """
    Match the pattern: contiguous -> unsqueeze(-1) -> unfold -> transpose -> reshape -> reshape
    This pattern is used for im2col-style grouped 1D convolution preparation.
    """
    tmp_0 = in_0.contiguous()
    tmp_1 = tmp_0.unsqueeze(-1)
    tmp_2 = torch.nn.functional.unfold(tmp_1, kernel_size=[9, 1], dilation=1, padding=[4, 0], stride=1)
    tmp_3 = tmp_2.transpose(1, 2)
    tmp_4 = tmp_3.reshape(1, -1, 16, 9)
    tmp_5 = torch.reshape(tmp_4, [-1, 8, 9])
    return tmp_5


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def fused_unfold_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    channels,
    length,
    padding: tl.constexpr,
    kernel_size: tl.constexpr,
    channels_per_group: tl.constexpr,
    num_groups: tl.constexpr,
    output_length: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for im2col-style grouped 1D convolution preparation.
    
    Args:
        input_ptr: Input tensor [batch_size, channels, length]
        output_ptr: Output tensor [num_groups * output_length, channels_per_group, kernel_size]
        batch_size: Batch size (should be 1)
        channels: Number of channels
        length: Input length
        padding: Padding size (4)
        kernel_size: Kernel size (9)
        channels_per_group: Number of channels per group
        num_groups: Number of groups
        output_length: Output length = length + 2*padding - kernel_size + 1
    """
    # Calculate global program ID
    pid = tl.program_id(0)
    
    # Calculate output coordinates
    # output_shape = [num_groups * output_length, channels_per_group, kernel_size]
    # Flatten first two dims: index = group_idx * output_length + out_idx
    group_idx = pid // output_length
    out_idx = pid % output_length
    
    # Channel offset within the group
    channel_offset = tl.arange(0, channels_per_group)
    
    # Kernel offset
    kernel_offset = tl.arange(0, kernel_size)
    
    # Compute input positions for each channel in the group
    # Global channel = group_idx * channels_per_group + channel_offset
    global_channel = group_idx * channels_per_group + channel_offset
    
    # Compute input position for each kernel element
    # input_idx = out_idx + kernel_offset - padding
    input_pos = out_idx + kernel_offset - padding
    
    # Create masks
    batch_idx = 0  # batch size is 1
    
    # Mask for valid channels and positions
    channel_mask = global_channel < channels
    pos_mask = (input_pos >= 0) & (input_pos < length)
    valid_mask = channel_mask & pos_mask
    
    # Load from input [batch, channels, length]
    input_offsets = (
        batch_idx * channels * length +
        global_channel * length +
        input_pos
    )
    
    # Load values (0 where padded)
    values = tl.load(input_ptr + input_offsets, mask=valid_mask, other=0.0)
    
    # Store to output [num_groups * output_length, channels_per_group, kernel_size]
    # Compute output offset
    output_offsets = (
        pid * channels_per_group * kernel_size +
        channel_offset * kernel_size +
        kernel_offset
    )
    
    tl.store(output_ptr + output_offsets, values)


@torch.fx.wrap
def fused_unfold_wrapper(input_tensor):
    """
    Wrapper for the fused unfold kernel.
    Handles shape inference and kernel launch.
    """
    batch_size, channels, length = input_tensor.shape
    kernel_size = 9
    padding = 4
    stride = 1
    channels_per_group = channels // 8  # 16 // 8 = 2 for tiny, 384 // 64 = 6 for base
    num_groups = 8  # 16 // 2 = 8 for tiny, 384 // 6 for base
    
    # Calculate output length: length + 2*padding - kernel_size + 1
    output_length = length + 2 * padding - kernel_size + 1
    
    # Output shape: [num_groups * output_length, channels_per_group, kernel_size]
    out_batch = num_groups * output_length
    out_channels = channels_per_group
    out_length = kernel_size
    
    # Allocate output
    output = torch.empty(
        (out_batch, out_channels, out_length),
        dtype=input_tensor.dtype,
        device=input_tensor.device
    )
    
    # Calculate grid
    totalPrograms = out_batch  # Each program handles one [channels_per_group, kernel_size] slice
    
    # Launch kernel
    fused_unfold_kernel[(totalPrograms,)](
        input_tensor,
        output,
        batch_size,
        channels,
        length,
        padding,
        kernel_size,
        channels_per_group,
        num_groups,
        output_length,
        BLOCK_SIZE=64,
    )
    
    return output


def replacement_func():
    return fused_unfold_wrapper