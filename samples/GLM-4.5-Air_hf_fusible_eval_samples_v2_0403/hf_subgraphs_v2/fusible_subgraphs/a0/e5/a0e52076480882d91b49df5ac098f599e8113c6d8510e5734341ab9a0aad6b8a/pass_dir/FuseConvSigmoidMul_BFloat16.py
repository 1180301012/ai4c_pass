import torch
import triton
import triton.language as tl

def pattern(x_se_2, conv_weight, conv_bias, out_52):
    """Pattern for: conv2d -> sigmoid -> element-wise multiplication"""
    conv2d = torch.conv2d(x_se_2, conv_weight, conv_bias, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.sigmoid()
    tmp_4 = out_52 * tmp_3
    return tmp_4

def replacement_args(x_se_2, conv_weight, conv_bias, out_52):
    return (x_se_2, conv_weight, conv_bias, out_52)

@triton.jit
def fused_conv_sigmoid_mul_kernel(
    x_ptr,      # [1, 64, 1, 1]
    w_ptr,      # [1024, 64, 1, 1] 
    b_ptr,      # [1024]
    y_ptr,      # [1, 1024, 7, 7]
    out_ptr,    # [1, 1024, 7, 7]
    
    # Input shapes
    C_out, H_out, W_out,
    C_in,
    
    # Strides
    x_stride_c,
    w_stride_c, w_stride_n,
    b_stride_c,
    y_stride_c, y_stride_h, y_stride_w,
    out_stride_c, out_stride_h, out_stride_w,
    
    BLOCK_SIZE: tl.constexpr,
):
    # Initialize program IDs (simplified: one thread per output channel only)
    pid_c = tl.program_id(0)  # Output channel
    pid_h = tl.program_id(1)  # Output height 
    pid_w = tl.program_id(2)  # Output width
    
    # Handle out-of-bounds channels
    if pid_c >= C_out:
        return
    
    # Load convolution weights for this output channel: use constexpr for fixed-size arange
    weights = tl.load(w_ptr + pid_c * w_stride_c + tl.arange(0, 64) * w_stride_n)
    
    # Load bias for this output channel
    bias = tl.load(b_ptr + pid_c * b_stride_c)
    
    # Load input x_se_2: [64]
    x = tl.load(x_ptr + tl.arange(0, 64) * x_stride_c)
    
    # Compute convolution: sum(weights * x) + bias
    conv_value = tl.sum(weights * x) + bias
    
    # Apply sigmoid - always use fp32 for exp then cast back
    conv_value_fp32 = tl.cast(conv_value, tl.float32)
    sigmoid_val_fp32 = 1.0 / (1.0 + tl.exp(-conv_value_fp32))
    sigmoid_val = tl.cast(sigmoid_val_fp32, conv_value.dtype)
    
    # Broadcast sigmoid to match spatial dimensions: load and multiply with y
    # We process spatial positions in blocks for vectorization
    spatial_pos = tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid spatial positions
    mask = spatial_pos < (H_out * W_out)
    
    # Convert linear spatial index to 2D coordinates
    h_coords = spatial_pos // W_out
    w_coords = spatial_pos % W_out
    
    # Calculate offsets in y and output tensors
    y_offsets = pid_c * y_stride_c + h_coords * y_stride_h + w_coords * y_stride_w
    out_offsets = pid_c * out_stride_c + h_coords * out_stride_h + w_coords * out_stride_w
    
    # Load y values with bounds checking
    y_values = tl.load(y_ptr + y_offsets, mask=mask, other=0.0)
    
    # Apply element-wise multiplication with sigmoid (broadcasted)
    out_values = y_values * sigmoid_val
    
    # Store results
    tl.store(out_ptr + out_offsets, out_values, mask=mask)

@torch.fx.wrap
def fused_conv_sigmoid_mul(x_se_2, conv_weight, conv_bias, out_52):
    # Get tensor shapes
    N, C_in, H_in, W_in = x_se_2.shape
    _, C_out, _, _ = conv_weight.shape
    _, _, H_out, W_out = out_52.shape
    
    # Calculate strides (simplified for our specific tensor shapes)
    x_stride_c = x_se_2.stride(1)  # Only need channel stride for [1, 64, 1, 1]
    
    w_stride_c = conv_weight.stride(0)  # Output channel stride
    w_stride_n = conv_weight.stride(1)  # Input channel stride
    
    b_stride_c = conv_bias.stride(0)  # Bias stride
    
    y_stride_c = out_52.stride(1)  # Output channel stride
    y_stride_h = out_52.stride(2)  # Height stride
    y_stride_w = out_52.stride(3)  # Width stride
    
    # Create output tensor
    output = torch.empty_like(out_52)
    
    # Set output strides
    out_stride_c = output.stride(1)
    out_stride_h = output.stride(2)
    out_stride_w = output.stride(3)
    
    BLOCK_SIZE = 128  # Number of elements to process per thread
    
    # Calculate grid sizes (simplified - each thread handles one channel and processes multiple spatial positions)
    # Since 7x7=49 spatial positions, we use BLOCK_SIZE=64 to ensure we cover all positions
    num_c = C_out  # One thread per output channel
    num_h = 1      # Single spatial dimension for coordinate conversion
    num_w = 1      # Single spatial dimension for coordinate conversion
    
    # Launch kernel with 3D grid
    fused_conv_sigmoid_mul_kernel[(num_c, num_h, num_w)](
        x_se_2, conv_weight, conv_bias, out_52, output,
        C_out, H_out, W_out,
        C_in,
        x_stride_c,
        w_stride_c, w_stride_n,
        b_stride_c,
        y_stride_c, y_stride_h, y_stride_w,
        out_stride_c, out_stride_h, out_stride_w,
        BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return fused_conv_sigmoid_mul