import torch
import triton
import triton.language as tl

def pattern(args):
    input, weight, bias = args
    conv = torch.conv2d(input, weight, bias, stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=128)
    gelu = torch.nn.functional.gelu(conv)
    return gelu

def replacement_args(args):
    return args

@triton.jit
def fused_conv3x3_gelu_kernel_128(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_channels, in_height, in_width,
    out_channels, kernel_h, kernel_w,
    stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
    groups, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Determine range of output channels per program
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    m_mask = m_offsets < out_channels
    m_offsets = m_offsets[m_mask]
    
    if len(m_offsets) == 0:
        return
    
    # For grouped convolution, calculate which groups these output channels belong to
    channels_per_group = out_channels // groups
    group_ids = m_offsets // channels_per_group
    local_channel_ids = m_offsets % channels_per_group
    
    # Calculate spatial locations each program handles
    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    n_mask = n_offsets < in_height * in_width
    n_offsets = n_offsets[n_mask]
    n_coords_h = n_offsets // in_width
    n_coords_w = n_offsets % in_width
    
    # Load bias vector for the assigned output channels
    bias_vector = tl.load(bias_ptr + m_offsets, mask=m_mask)
    
    # Initialize accumulator
    accumulator = tl.zeros((len(m_offsets), len(n_offsets)), dtype=tl.float32)
    
    # Each group handles independent computation, process one group at a time
    for group_idx in group_ids:
        # Find all m_offsets that belong to this group
        group_positions = (group_ids == group_idx)
        group_m_offsets = m_offsets[group_positions]
        group_local_channels = local_channel_ids[group_positions]
        
        # Calculate which output channels we're computing for this group
        output_group_start = group_idx * channels_per_group
        local_positions_in_group = group_local_channels
        
        # For each kernel position
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                # Calculate input spatial location for each output location
                input_h = n_coords_h * stride_h + kh * dilation_h - pad_h
                input_w = n_coords_w * stride_w + kw * dilation_w - pad_w
                
                # Check bounds and create valid mask
                valid_h = (input_h >= 0) & (input_h < in_height)
                valid_w = (input_w >= 0) & (input_w < in_width)
                valid_mask = valid_h & valid_w
                
                # Skip if no valid positions
                if not tl.any(valid_mask):
                    continue
                
                # Number of input channels per group
                in_channels_per_group = in_channels // groups
                
                # Calculate input pointer for this group
                base_input_ptr = input_ptr + (
                    input_h[:, None] * in_width * in_channels +  # H position
                    input_w[:, None] * in_channels +            # W position  
                    group_idx * in_channels_per_group           # Base offset for this group
                )
                
                # Calculate weight pointer for this group and kernel position
                base_weight_ptr = weight_ptr + (
                    group_m_offsets[:, None] * kernel_h * kernel_w * in_channels_per_group +
                    kh * kernel_w * in_channels_per_group +
                    kw * in_channels_per_group
                )
                
                # Load input for valid positions
                input_values = tl.load(base_input_ptr, mask=valid_mask[:, None], other=0.0)
                
                # Load weights and compute
                for k in range(0, in_channels_per_group, BLOCK_SIZE_K):
                    k_offsets = tl.arange(0, BLOCK_SIZE_K)
                    k_mask = k + k_offsets < in_channels_per_group
                    k_offsets_valid = k + k_offsets
                    
                    # Load input slice
                    input_slice = input_values[..., k_offsets_valid] if len(input_values.shape) > 2 else input_values
                    
                    # Load weight slice
                    weight_ptr_slice = base_weight_ptr[..., k_offsets_valid]
                    weight_slice = tl.load(weight_ptr_slice, mask=k_offsets_valid >= 0, other=0.0)
                    
                    # Matrix multiply and accumulate
                    accumulator[group_positions, :, None] += input_slice * weight_slice[None, ...]
    
    # Apply GELU activation
    # Using the approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = tl.sqrt(2.0 / tl.pi)
    gelu_cubic = 0.044715
    gelu_activation = 0.5 * accumulator * (1.0 + tl.tanh(sqrt_2_over_pi * (accumulator + gelu_cubic * accumulator * accumulator * accumulator)))
    
    # Add bias
    output_with_bias = gelu_activation + bias_vector[:, None]
    
    # Store result
    output_ptrs = output_ptr + (m_offsets[:, None] * in_height * in_width + n_offsets[None, :])
    tl.store(output_ptrs, output_with_bias, mask=m_offsets[:, None] < out_channels)

@torch.fx.wrap
def fused_conv3x3_gelu_128(input, weight, bias):
    batch_size, in_channels, in_height, in_width = input.shape
    out_channels, kernel_h, kernel_w = weight.shape[:3]
    groups = 128  # Fixed for this pass
    
    # Output dimensions for 3x3 conv with padding=1, stride=1
    out_height = ((in_height + 2 * pad_h - kernel_h) // stride_h) + 1
    out_width = ((in_width + 2 * pad_w - kernel_w) // stride_w) + 1
    
    # Create output tensor
    output = torch.empty((batch_size, out_channels, out_height, out_width), device=input.device, dtype=input.dtype)
    
    # Optimize block sizes for GPU with grouped convolution
    BLOCK_SIZE_M = 128   # Output channels per program (must divide group_size)
    BLOCK_SIZE_N = 1024  # Spatial locations per program
    BLOCK_SIZE_K = 32    # Input channels per loop
    
    # Calculate grid dimensions
    num_programs_m = (out_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_programs_n = (in_height * in_width + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel
    fused_conv3x3_gelu_kernel_128[(num_programs_m, num_programs_n)](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        in_height=in_height,
        in_width=in_width,
        out_channels=out_channels,
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        stride_h=1, stride_w=1,
        pad_h=1, pad_w=1,
        dilation_h=1, dilation_w=1,
        groups=groups,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return output

def replacement_func():
    return fused_conv3x3_gelu_128