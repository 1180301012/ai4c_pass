import torch
import triton
import triton.language as tl

# Pure Triton implementation for 1x1 conv + reshape
# Uses 1D grid approach to avoid dynamic tensor indexing issues

BLOCK_SIZE = tl.constexpr(256)  # Process this many output elements per thread

@triton.jit
def conv_reshape_kernel(
    input_ptr,      # [B, C_in, H, W]
    weight_ptr,     # [C_out, C_in, 1, 1]
    bias_ptr,       # [C_out]
    output_ptr,     # [B, C_out, H*W] - contiguous
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    input_stride_B,
    input_stride_C,
    input_stride_H,
    input_stride_W,
    weight_stride_CO,
    weight_stride_CI,
    output_size,  # total number of output elements
):
    """
    Fused 1x1 conv + reshape kernel.
    Uses 1D grid: each thread processes BLOCK_SIZE output elements.
    """
    # Get starting index for this thread
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_size
    
    # Compute (batch, out_channel, spatial) indices from linear offset
    # output_size = batch_size * out_channels * spatial_size
    spatial_size = height * width
    
    # Unpack: offset -> (batch, out_channel, spatial_idx)
    tmp = offsets // (out_channels * spatial_size)
    batch_idx = tmp
    remaining = offsets % (out_channels * spatial_size)
    out_channel_idx = remaining // spatial_size
    spatial_idx = remaining % spatial_size
    
    # Compute h, w from spatial_idx
    h_idx = spatial_idx // width
    w_idx = spatial_idx % width
    
    # Initialize accumulator: [BLOCK_SIZE]
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Load bias
    bias_vals = tl.load(bias_ptr + out_channel_idx, mask=mask, other=0.0)
    
    # Reduction over input channels
    for c_in in range(in_channels):
        # Load weight: [C_out, C_in, 1, 1]
        w_linear_idx = out_channel_idx * weight_stride_CO + c_in * weight_stride_CI
        weight_vals = tl.load(weight_ptr + w_linear_idx, mask=mask, other=0.0)
        
        # Load input: [B, C_in, H, W]
        # Linear index: b*stride_B + c_in*stride_C + h*stride_H + w*stride_W
        input_linear_idx = (batch_idx * input_stride_B + 
                           c_in * input_stride_C + 
                           h_idx * input_stride_H + 
                           w_idx * input_stride_W)
        input_vals = tl.load(input_ptr + input_linear_idx, mask=mask, other=0.0)
        
        # Accumulate: acc += weight * input
        acc += weight_vals * input_vals
    
    # Add bias
    acc = acc + bias_vals
    
    # Store output: [B, C_out, spatial_size]
    # Linear output index: b * C_out * spatial_size + c_out * spatial_size + spatial_idx
    output_linear_idx = (batch_idx * out_channels * spatial_size + 
                        out_channel_idx * spatial_size + 
                        spatial_idx)
    tl.store(output_ptr + output_linear_idx, acc, mask=mask)

@torch.fx.wrap
def conv2d_reshape_fused(in_0, in_1, in_2):
    """
    Fused conv2d + reshape operation using pure Triton.
    
    in_0: bias tensor [17]
    in_1: weight tensor [17, 256, 1, 1] 
    in_2: input tensor [batch, 256, 64, 64]
    
    Returns: [batch, 17, 4096]
    """
    # Get shapes
    batch_size, in_channels, height, width = in_2.shape
    out_channels = in_1.shape[0]  # 17
    spatial_size = height * width  # 64 * 64 = 4096
    
    # Total output elements
    output_size = batch_size * out_channels * spatial_size
    
    # Allocate output: [B, C_out, H*W] = [B, 17, 4096]
    output = torch.empty((batch_size, out_channels, spatial_size), 
                         dtype=in_2.dtype, device=in_2.device)
    
    # Calculate grid
    num_programs = (output_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    conv_reshape_kernel[(num_programs,)](
        input_ptr=in_2,
        weight_ptr=in_1,
        bias_ptr=in_0,
        output_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        height=height,
        width=width,
        input_stride_B=in_2.stride(0),
        input_stride_C=in_2.stride(1),
        input_stride_H=in_2.stride(2),
        input_stride_W=in_2.stride(3),
        weight_stride_CO=in_1.stride(0),
        weight_stride_CI=in_1.stride(1),
        output_size=output_size,
    )
    
    return output

def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d * 1.0
    tmp_4 = tmp_3.reshape(-1, 17, 4096)
    return tmp_4

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

def replacement_func():
    return conv2d_reshape_fused