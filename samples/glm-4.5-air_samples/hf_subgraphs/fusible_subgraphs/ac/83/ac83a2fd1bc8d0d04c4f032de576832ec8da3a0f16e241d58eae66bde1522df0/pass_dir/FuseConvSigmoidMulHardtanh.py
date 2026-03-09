import torch
import triton
import triton.language as tl
import math

def pattern(bias, weights, in_2, in_3):
    tmp_2 = torch.conv2d(in_3, weights, bias, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = tmp_2.sigmoid()
    tmp_4 = in_2 * tmp_3
    tmp_5 = torch.nn.functional.hardtanh(tmp_4, 0.0, 6.0, False)
    return tmp_5

def replacement_args(bias, weights, in_2, in_3):
    return (bias, weights, in_2, in_3)

@triton.jit
def fused_conv_activation_kernel(
    bias_ptr,
    weights_ptr, 
    in_2_ptr,
    in_3_ptr,
    out_ptr,
    batch_size,
    out_channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID for 1D grid
    pid = tl.program_id(0)
    
    # Create range offsets within block (power of 2)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < batch_size * out_channels * height * width
    
    # Skip invalid work items early for performance
    if ~mask:
        return
    
    # Extract coordinates
    linear_idx = offs
    batch_idx = linear_idx // (out_channels * height * width)
    channel_idx = (linear_idx // (height * width)) % out_channels
    spatial_idx = linear_idx % (height * width)
    
    # Validate indices
    if batch_idx >= batch_size or channel_idx >= out_channels or spatial_idx >= height * width:
        return
    
    # Load bias for this channel
    bias = tl.load(bias_ptr + channel_idx)
    
    # Load weights for this channel - loop over the 19 input channels
    conv_sum = bias  # Start with bias
    for ic in range(19):
        # Load weight for this input/output channel pair
        weight_offset = channel_idx * 19 + ic
        weight = tl.load(weights_ptr + weight_offset)
        
        # Load input for this batch and input channel
        input_offset = batch_idx * 19 + ic
        input_val = tl.load(in_3_ptr + input_offset)
        
        # Accumulate dot product
        conv_sum += input_val * weight
    
    # Apply sigmoid activation
    sigmoid_out = 1.0 / (1.0 + tl.exp(-conv_sum))
    
    # Load multiplier tensor for this spatial position
    multiplier_offset = linear_idx
    multiplier = tl.load(in_2_ptr + multiplier_offset)
    
    # Element-wise multiplication with sigmoid output
    mul_out = sigmoid_out * multiplier
    
    # Apply hardtanh activation with bounds [0, 6]
    hardtanh_out = tl.maximum(0.0, tl.minimum(mul_out, 6.0))
    
    # Store result
    tl.store(out_ptr + linear_idx, hardtanh_out)

@torch.fx.wrap
def fused_convolution_activation(bias, weights, in_2, in_3):
    output_shape = in_2.shape
    batch_size, out_channels, height, width = in_2.shape[0], in_2.shape[1], in_2.shape[2], in_2.shape[3]
    
    # Use power-of-2 block size for efficiency
    BLOCK_SIZE_BC = 256  # Power of 2 for combined batch-channel spatial dimension
    
    # Calculate grid dimensions
    total_elements = batch_size * out_channels * height * width
    num_blocks = math.ceil(total_elements / BLOCK_SIZE_BC)
    
    out = torch.empty_like(in_2)
    
    fused_conv_activation_kernel[(num_blocks,)](
        bias_ptr=bias,
        weights_ptr=weights,
        in_2_ptr=in_2,
        in_3_ptr=in_3,
        out_ptr=out,
        batch_size=batch_size,
        out_channels=out_channels,
        height=height,
        width=width,
        BLOCK_SIZE_BC=BLOCK_SIZE_BC,
    )
    
    return out

def replacement_func():
    return fused_convolution_activation