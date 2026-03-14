import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1, in_2, in_3):
    # Exactly match the computation structure from model.py
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_3, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_1 = tmp_0 = None
    tmp_3 = tmp_2 + 1.0
    tmp_2 = None
    tmp_4 = tmp_3 / 2.0
    tmp_3 = None
    tmp_5 = tmp_4.clamp_(0.0, 1.0)
    tmp_4 = None
    tmp_6 = in_2 * tmp_5
    tmp_5 = None
    return (tmp_6,)

# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

# Optimized fused kernel
@triton.jit
def fused_conv_add_div_clamp_mul_kernel(
    bias_ptr,
    weight_ptr, 
    scale_ptr,  # in_2 for multiplication
    input_ptr,  # in_3 for conv input
    out_ptr,
    n_batch,
    n_channels_out,
    n_channels_in,
    height,
    width,
    scale_stride_0: tl.constexpr,
    scale_stride_1: tl.constexpr,
    scale_stride_2: tl.constexpr,
    input_stride_0: tl.constexpr,
    input_stride_1: tl.constexpr,
    input_stride_2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Triton kernel for fused: Conv2D -> (x + 1.0) / 2.0 -> clamp(0, 1) -> * scale
    
    # Program ID (1D grid)
    pid = tl.program_id(0)
    
    # Create flattened offsets
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Only proceed within bounds
    mask = offset < (n_batch * n_channels_out * height * width)
    
    # Convert flattened offset to [batch, channel, height, width] indices
    # Total elements per spatial location: n_channels_out
    # Total elements per batch: n_channels_out * height * width
    spatial_offset = offset
    batch_idx = spatial_offset // (n_channels_out * height * width)
    remaining = spatial_offset % (n_channels_out * height * width)
    channel_idx = remaining // (height * width)
    spatial_idx = remaining % (height * width)
    
    # Load bias for this output channel (with masking)
    channel_mask = channel_idx < n_channels_out
    bias_value = tl.load(bias_ptr + channel_idx, mask=channel_mask, other=0.0).to(tl.float32)
    
    # Load scale tensor value [batch, n_channels_out, height, width]
    scale_offset = batch_idx * scale_stride_0 + channel_idx * scale_stride_1 + spatial_idx * scale_stride_2
    scale_value = tl.load(scale_ptr + scale_offset, mask=mask, other=0.0).to(tl.float32)
    
    # Initialize result with bias
    result = bias_value
    
    # Perform 1x1 convolution for this position
    for k in range(n_channels_in):
        # Load weight [n_channels_out, n_channels_in] for spatial position [h, w] = [0, 0]
        weight_offset = channel_idx * n_channels_in + k
        weight_mask = (channel_idx < n_channels_out) & (k < n_channels_in)
        weight_value = tl.load(weight_ptr + weight_offset, mask=weight_mask, other=0.0).to(tl.float32)
        
        # Load input [batch, n_channels_in, height, width] - always [h, w] = [0, 0] for this conv
        input_offset = batch_idx * input_stride_0 + k * input_stride_1 + spatial_idx * input_stride_2
        input_value = tl.load(input_ptr + input_offset, mask=mask, other=0.0).to(tl.float32)
        
        # Multiply and accumulate
        result += weight_value * input_value
    
    # Fused operations: add 1.0, divide by 2.0, clamp to [0, 1]
    result = result + 1.0
    result = result * 0.5  # More efficient than division
    result = tl.maximum(result, 0.0)
    result = tl.minimum(result, 1.0)
    
    # Multiply by scale
    result = result * scale_value
    
    # Store output (flatten back to 1D for store)
    output_offset = batch_idx * (n_channels_out * height * width) + channel_idx * (height * width) + spatial_idx
    tl.store(out_ptr + output_offset, result, mask=mask)

@torch.fx.wrap
def fused_conv_add_div_clamp_mul(in_0, in_1, in_2, in_3):
    """Fused kernel combining Conv2D + add + div + clamp + mul operations"""
    
    # Get tensor shapes
    bias_shape = in_0.shape  # [400]
    weight_shape = in_1.shape  # [400, 100, 1, 1]
    scale_shape = in_2.shape  # [N, 400, H, W]
    input_shape = in_3.shape  # [N, 100, 1, 1]
    
    # Extract dimensions
    n_batch, n_channels_in, h_in, w_in = input_shape
    n_channels_out = bias_shape[0]  # 400
    h_out = h_in  # No padding, same stride/dilation
    w_out = w_in
    
    # Launch Triton kernel
    out = torch.empty((n_batch, n_channels_out, h_out, w_out), dtype=torch.float32, device=in_0.device)
    
    # Optimized block sizes (using 1D grid to avoid dimension mismatch)
    BLOCK_SIZE = 256
    
    # Calculate strides for the tensors
    scale_stride_0 = in_2.stride(0)  # stride between N dimension
    scale_stride_1 = in_2.stride(1)  # stride between C dimension  
    scale_stride_2 = in_2.stride(2)  # stride between H dimension
    
    input_stride_0 = in_3.stride(0)  # stride between N dimension
    input_stride_1 = in_3.stride(1)  # stride between C dimension  
    input_stride_2 = in_3.stride(2)  # stride between H dimension
    
    # Calculate total elements and grid size (1D grid)
    total_elements = n_batch * n_channels_out * h_out * w_out
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel with 1D grid
    fused_conv_add_div_clamp_mul_kernel[(
        grid_size,
    )](
        bias_ptr=in_0,
        weight_ptr=in_1,
        scale_ptr=in_2,
        input_ptr=in_3,
        out_ptr=out,
        n_batch=n_batch,
        n_channels_out=n_channels_out,
        n_channels_in=n_channels_in,
        height=h_out,
        width=w_out,
        scale_stride_0=scale_stride_0,
        scale_stride_1=scale_stride_1,
        scale_stride_2=scale_stride_2,
        input_stride_0=input_stride_0,
        input_stride_1=input_stride_1,
        input_stride_2=input_stride_2,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function
def replacement_func():
    return fused_conv_add_div_clamp_mul