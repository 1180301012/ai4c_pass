import torch
import triton
import triton.language as tl

def pattern(args):
    input, weight, bias = args
    conv = torch.conv2d(input, weight, bias, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=256)
    gelu = torch.nn.functional.gelu(conv)
    return gelu

def replacement_args(args):
    return args

@triton.jit
def fused_conv1x1_gelu_kernel_256(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_channels, in_height, in_width,
    out_channels, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
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
        
        # Number of input channels per group (for 1x1 conv, in_channels == out_channels usually)
        in_channels_per_group = in_channels // groups
        
        # For 1x1 convolution, we can process all kernel positions at once
        # Since kernel is 1x1, kh=0, kw=0
        kh, kw = 0, 0
        
        # Calculate input spatial location (no change for 1x1 conv with stride=1)
        input_h = n_coords_h * stride_h + kh * dilation_h - pad_h
        input_w = n_coords_w * stride_w + kw * dilation_w - pad_w
        
        # Check bounds and create valid mask
        valid_h = (input_h >= 0) & (input_h < in_height)
        valid_w = (input_w >= 0) & (input_w < in_width)
        valid_mask = valid_h & valid_w
        
        # Skip if no valid spatial positions
        if not tl.any(valid_mask):
            continue
        
        # Calculate input pointer for this group at all spatial positions
        base_input_ptr = input_ptr + (
            input_h[:, None] * in_width * in_channels +    # H position (expanded to all W)
            input_w[:, None] * in_channels +              # W position (expanded to all groups)
            group_idx * in_channels_per_group             # Base offset for this group
        )
        
        # Load input for all positions in this group
        input_values = tl.load(base_input_ptr, mask=valid_mask[:, None], other=0.0)
        
        # Process input channels in chunks
        for k in range(0, in_channels_per_group, BLOCK_SIZE_K):
            k_offsets = tl.arange(0, BLOCK_SIZE_K)
            k_mask = k + k_offsets < in_channels_per_group
            k_offsets_valid = k + k_offsets[k_mask]
            
            # Load weight slice for this group and input channels
            weight_ptr_slice = weight_ptr + (
                group_m_offsets[:, None] * in_channels_per_group + 
                k_offsets_valid[None, :]
            )
            weight_slice = tl.load(weight_ptr_slice, mask=True, other=0.0)
            
            # Matrix multiply: input * weight for this group
            accumulator[group_positions, :] += input_values[..., k_offsets_valid[None, :]] * weight_slice[None, ...]
    
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
def fused_conv1x1_gelu_256(input, weight, bias):
    batch_size, in_channels, in_height, in_width = input.shape
    out_channels, _, _ = weight.shape[:3]
    groups = 256  # Fixed for this pass
    
    # For 1x1 conv with padding=0, stride=1: output size same as input
    out_height = in_height
    out_width = in_width
    
    # Create output tensor
    output = torch.empty((batch_size, out_channels, out_height, out_width), device=input.device, dtype=input.dtype)
    
    # Optimize block sizes for 1x1 grouped convolution
    BLOCK_SIZE_M = 128   # Output channels per program (must be multiple of group_size)
    BLOCK_SIZE_N = 2048  # Spatial locations per program
    BLOCK_SIZE_K = 64    # Input channels per loop
    
    # Calculate grid dimensions
    num_programs_m = (out_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_programs_n = (in_height * in_width + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel
    fused_conv1x1_gelu_kernel_256[(num_programs_m, num_programs_n)](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        in_height=in_height,
        in_width=in_width,
        out_channels=out_channels,
        stride_h=1, stride_w=1,
        pad_h=0, pad_w=0,
        dilation_h=1, dilation_w=1,
        groups=groups,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return output

def replacement_func():
    return fused_conv1x1_gelu_256