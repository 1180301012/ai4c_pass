import torch
import triton
import triton.language as tl

# Pattern matching function - matches the sequence from conv3d through transpose
def pattern(in_3, in_1, in_0):
    conv3d = torch.conv3d(in_3, in_1, in_0, (2, 16, 16), (0, 0, 0), (1, 1, 1), 1)
    tmp_4 = conv3d.flatten(2)
    tmp_5 = tmp_4.transpose(1, 2)
    return tmp_5

# Argument extraction function
def replacement_args(in_3, in_1, in_0):
    return (in_3, in_1, in_0)

# Optimized kernel that fuses conv3d, flatten, and transpose
@triton.jit
def fused_conv3d_flatten_transpose_kernel(
    input_ptr,      # in_3: [1, 3, 16, 224, 224]
    weight_ptr,     # in_1: [768, 3, 2, 16, 16] 
    bias_ptr,       # in_0: [768]
    output_ptr,     # tmp_5: [1, 655335, 768]
    
    # Input dimensions
    batch_size,         # 1
    in_channels,        # 3
    input_d, input_h, input_w,  # 16, 224, 224
    
    # Weight dimensions
    out_channels,       # 768
    kernel_d, kernel_h, kernel_w,  # 2, 16, 16
    
    # Output dimensions after convolution
    output_d, output_h, output_w,  # 15, 209, 209
    
    # Flattened dimensions
    flatten_dim,        # 15*209*209 = 655335
    
    # Convolution parameters
    stride_d, stride_h, stride_w,  # 2, 16, 16
    padding_d, padding_h, padding_w,  # 0, 0, 0
    dilation,           # 1
    groups,             # 1
    
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Get program IDs for output tensor [1, 655335, 768]
    pid_flat = tl.program_id(1)  # flattened spatial dimension  
    pid_channel = tl.program_id(2)  # output channel
    
    # Calculate output position in flattened space
    flat_pos = pid_flat * BLOCK_M + tl.arange(0, BLOCK_M)
    
    # Convert flat position back to 3D spatial coordinates
    spatial_pos = flat_pos
    w_pos = spatial_pos % output_w
    h_pos = (spatial_pos // output_w) % output_h  
    d_pos = (spatial_pos // (output_w * output_h)) % output_d
    
    # Create masks
    mask_flat = flat_pos < flatten_dim
    mask_channel = pid_channel < out_channels
    
    # Initialize accumulator 
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    # Calculate output channel range for this program
    current_channel = pid_channel * BLOCK_N 
    channel_mask = current_channel < out_channels
    
    # Simplified 3D convolution kernel
    # Process one output channel at a time for simplicity
    if current_channel < out_channels:
        # Iterate over input channels
        for in_c in range(in_channels):
            # Get weight pointer for this input channel and output channel
            weight_idx = current_channel * in_channels * kernel_d * kernel_h * kernel_w + \
                        in_c * kernel_d * kernel_h * kernel_w
            
            # Iterate over kernel spatial positions
            for kd in range(kernel_d):
                for kh in range(kernel_h):
                    for kw in range(kernel_w):
                        # Calculate input position
                        input_d_pos = d_pos * stride_d + kd - padding_d
                        input_h_pos = h_pos * stride_h + kh - padding_h
                        input_w_pos = w_pos * stride_w + kw - padding_w
                        
                        # Create input mask for valid positions
                        input_mask = ((input_d_pos >= 0) & (input_d_pos < input_d) &
                                     (input_h_pos >= 0) & (input_h_pos < input_h) &
                                     (input_w_pos >= 0) & (input_w_pos < input_w))
                        
                        # Load input data (will automatically use mask and pad with 0.0)
                        input_base = input_ptr + (0 * in_channels * input_d * input_h * input_w +
                                                 in_c * input_d * input_h * input_w +
                                                 input_d_pos * input_h * input_w +
                                                 input_h_pos * input_w + input_w_pos)
                        input_data = tl.load(input_base, mask=input_mask, other=0.0)
                        
                        # Load weight data (no mask needed since weights are always valid)
                        weight_base = weight_ptr + weight_idx + \
                                    (kd * kernel_h * kernel_w + kh * kernel_w + kw)
                        weight_data = tl.load(weight_base)
                        
                        # Accumulate
                        acc += input_data * weight_data
        
        # Add bias
        bias_data = tl.load(bias_ptr + current_channel)
        acc += bias_data
    
    # Store output in [1, flatten_dim, out_channels] layout
    output_base = output_ptr + (0 * flatten_dim * out_channels +
                               flat_pos * out_channels + current_channel)
    tl.store(output_base, acc, mask=(mask_flat & channel_mask))

@torch.fx.wrap
def fused_conv3d_flatten_transpose(in_3, in_1, in_0):
    # Get input shapes
    input_shape = in_3.shape  # [1, 3, 16, 224, 224]
    weight_shape = in_1.shape  # [768, 3, 2, 16, 16]
    bias_shape = in_0.shape   # [768]
    
    # Calculate output dimensions
    batch_size, in_channels, input_d, input_h, input_w = input_shape
    out_channels, kernel_d, kernel_h, kernel_w = weight_shape[0], weight_shape[2], weight_shape[3], weight_shape[4]
    
    # Convolution parameters
    stride_d, stride_h, stride_w = 2, 16, 16
    padding_d, padding_h, padding_w = 0, 0, 0
    dilation = 1
    groups = 1
    
    # Calculate output dimensions after convolution
    output_d = (input_d + 2 * padding_d - dilation * (kernel_d - 1) - 1) // stride_d + 1
    output_h = (input_h + 2 * padding_h - dilation * (kernel_h - 1) - 1) // stride_h + 1
    output_w = (input_w + 2 * padding_w - dilation * (kernel_w - 1) - 1) // stride_w + 1
    
    # Flattened spatial dimension
    flatten_dim = output_d * output_h * output_w  # 15 * 209 * 209 = 655335
    
    # Create output tensor [1, flatten_dim, out_channels]  
    output_shape = (batch_size, flatten_dim, out_channels)
    out = torch.empty(output_shape, dtype=in_3.dtype, device=in_3.device)
    
    # Block sizes for GPU utilization
    BLOCK_M = 128   # Process 128 flattened positions per program
    BLOCK_N = 64    # Process 64 output channels per program  
    
    # Grid dimensions
    num_batch = 1  # Only one batch
    num_flat = (flatten_dim + BLOCK_M - 1) // BLOCK_M
    num_channel = (out_channels + BLOCK_N - 1) // BLOCK_N
    
    # Launch kernel
    fused_conv3d_flatten_transpose_kernel[(num_batch, num_flat, num_channel)](
        input_ptr=in_3,
        weight_ptr=in_1,
        bias_ptr=in_0,
        output_ptr=out,
        batch_size=batch_size,
        in_channels=in_channels,
        input_d=input_d, input_h=input_h, input_w=input_w,
        out_channels=out_channels,
        kernel_d=kernel_d, kernel_h=kernel_h, kernel_w=kernel_w,
        output_d=output_d, output_h=output_h, output_w=output_w,
        flatten_dim=flatten_dim,
        stride_d=stride_d, stride_h=stride_h, stride_w=stride_w,
        padding_d=padding_d, padding_h=padding_h, padding_w=padding_w,
        dilation=dilation,
        groups=groups,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    
    return out

# Replacement function (returns function reference)
def replacement_func():
    return fused_conv3d_flatten_transpose