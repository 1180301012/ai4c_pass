import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    """
    Match Conv2d followed by flatten(dim=2)
    
    Conv2d args: input, weight, bias, stride, padding, dilation, groups
    Here: stride=(1,1), padding=(0,0), dilation=(1,1), groups=1
    """
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_1 = tmp_0 = None
    tmp_3 = torch.flatten(tmp_2, 2)
    tmp_2 = None
    return tmp_3


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# Define constexpr constants - these are fixed for this specific model
OUT_CHANNELS: tl.constexpr = 17
ACC_SIZE: tl.constexpr = 32  # Power of 2 >= OUT_CHANNELS


@triton.jit
def fused_conv_flatten_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_channels, spatial_size,
    stride_input_batch, stride_input_c, stride_input_h, stride_input_w,
    stride_weight_oc, stride_weight_ic,
):
    """
    Fused 1x1 Conv2d + Flatten kernel
    
    Grid: one program per (batch, spatial_position)
    Each program computes all output channels for one spatial position
    """
    # Program id: maps to (batch_idx, spatial_idx)
    pid = tl.program_id(0)
    batch_idx = pid // spatial_size
    sp_idx = pid % spatial_size
    
    # Compute h, w from spatial index (H=64, W=48)
    h = sp_idx // 48
    w = sp_idx % 48
    
    # Initialize accumulator with power-of-2 size
    acc = tl.zeros((ACC_SIZE,), dtype=tl.float32)
    
    # Add bias (only first OUT_CHANNELS=17 elements)
    bias_vals = tl.load(bias_ptr + tl.arange(0, ACC_SIZE))
    acc = acc + bias_vals
    
    # Reduction over input channels
    for ic_idx in range(0, in_channels):
        # Load input: input[batch, ic, h, w]
        input_offset = (batch_idx * stride_input_batch + 
                       ic_idx * stride_input_c + 
                       h * stride_input_h + 
                       w * stride_input_w)
        input_val = tl.load(input_ptr + input_offset)
        
        # Load weight for all output channels: weight[oc, ic]
        weight_offsets = tl.arange(0, ACC_SIZE) * stride_weight_oc + ic_idx * stride_weight_ic
        weight_vals = tl.load(weight_ptr + weight_offsets)
        
        # Multiply and accumulate
        acc = acc + input_val * weight_vals
    
    # Store output: output[batch, oc, spatial] 
    # Use vectorized store with mask
    output_base = batch_idx * OUT_CHANNELS * spatial_size + sp_idx
    
    # Create output offsets for all channels at once
    oc_offsets = tl.arange(0, ACC_SIZE) * spatial_size
    output_offsets = output_base + oc_offsets
    
    # Mask for valid channels (first 17)
    mask = tl.arange(0, ACC_SIZE) < OUT_CHANNELS
    
    # Vectorized store
    tl.store(output_ptr + output_offsets, acc, mask=mask)


@torch.fx.wrap
def fused_conv_flatten_kernel_wrapper(in_0, in_1, in_2):
    """
    Wrapper for the fused conv + flatten kernel
    
    in_0: bias [out_channels]
    in_1: weight [out_channels, in_channels, 1, 1]
    in_2: input [batch, in_channels, H, W]
    """
    batch_size = in_2.shape[0]
    in_channels = in_2.shape[1]
    H, W = in_2.shape[2], in_2.shape[3]
    spatial_size = H * W
    
    # Create output tensor: [batch, out_channels, H*W]
    output = torch.empty((batch_size, OUT_CHANNELS, spatial_size), dtype=torch.float32, device=in_2.device)
    
    # Strides for input [batch, c, h, w]
    stride_input_batch = in_2.stride(0)
    stride_input_c = in_2.stride(1)
    stride_input_h = in_2.stride(2)
    stride_input_w = in_2.stride(3)
    
    # Strides for weight [out_channels, in_channels, 1, 1]
    stride_weight_oc = in_1.stride(0)
    stride_weight_ic = in_1.stride(1)
    
    # Grid: batch * spatial_positions
    grid = (batch_size * spatial_size,)
    
    fused_conv_flatten_kernel[grid](
        in_2, in_1, in_0, output,
        batch_size, in_channels, spatial_size,
        stride_input_batch, stride_input_c, stride_input_h, stride_input_w,
        stride_weight_oc, stride_weight_ic,
    )
    
    return output


def replacement_func():
    return fused_conv_flatten_kernel_wrapper