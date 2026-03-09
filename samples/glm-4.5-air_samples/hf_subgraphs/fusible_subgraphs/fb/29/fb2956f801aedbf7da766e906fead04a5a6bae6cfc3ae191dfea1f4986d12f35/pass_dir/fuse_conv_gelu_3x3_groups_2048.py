import torch
import triton
import triton.language as tl

def pattern(args):
    input, weight, bias = args
    conv = torch.conv2d(input, weight, bias, stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=2048)
    gelu = torch.nn.functional.gelu(conv)
    return gelu

def replacement_args(args):
    return args

@triton.jit
def fused_conv3x3_gelu_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_channels, in_height, in_width,
    out_channels, kernel_h, kernel_w,
    stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
    groups, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Range of output channels per program
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    m_mask = m_offsets < out_channels
    m_offsets = m_offsets[m_mask]
    
    if len(m_offsets) == 0:
        return
    
    # Calculate spatial positions per program
    group_size = out_channels // groups
    group_base = m_offsets // group_size
    local_m_offsets = m_offsets % group_size
    
    # Spatial positions in output
    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    n_mask = n_offsets < in_height * in_width
    n_offsets = n_offsets[n_mask]
    n_coords_h = n_offsets // in_width
    n_coords_w = n_offsets % in_width
    
    # Load bias
    bias_vector = tl.load(bias_ptr + m_offsets, mask=m_mask)
    
    # Initialize accumulator
    accumulator = tl.zeros((len(m_offsets), len(n_offsets)), dtype=tl.float32)
    
    # Number of input channels per group
    channels_per_group = in_channels // groups
    
    # GELU constants
    sqrt_2_over_pi = tl.sqrt(2.0 / tl.pi)
    gelu_cubic_const = 0.044715
    
    # Main kernel computation
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            # Calculate input spatial positions for this kernel position
            input_h = n_coords_h * stride_h + kh * dilation_h - pad_h
            input_w = n_coords_w * stride_w + kw * dilation_w - pad_w
            
            # Only process valid input positions
            h_valid_mask = (input_h >= 0) & (input_h < in_height)
            w_valid_mask = (input_w >= 0) & (input_w < in_width)
            valid_mask = h_valid_mask & w_valid_mask
            
            # We handle boundary conditions by masking out invalid loads
            if not tl.any(valid_mask):
                continue
            
            # Load input patches for all groups simultaneously
            input_ptrs = input_ptr + (
                input_h[:, None] * in_width * in_channels + 
                input_w[:, None] * in_channels + 
                group_base[:, None] * channels_per_group
            )
            input_patches = tl.load(input_ptrs, mask=valid_mask[:, None] & (group_base[:, None] < groups), other=0.0)
            
            # Load weight patches
            weight_ptrs = weight_ptr + (
                m_offsets[:, None] * kernel_h * kernel_w * channels_per_group + 
                kh * kernel_w * channels_per_group + 
                kw * channels_per_group + 
                group_base[:, None] * channels_per_group
            )
            weight_patches = tl.load(weight_ptrs, mask=(group_base[:, None] < groups), other=0.0)
            
            # Matrix multiply: input_patches * weight_patches for the full batch of output channels
            for k in range(0, channels_per_group, BLOCK_SIZE_K):
                k_offsets = tl.arange(0, BLOCK_SIZE_K)
                k_mask = k + k_offsets < channels_per_group
                k_offsets_valid = k + k_offsets[k_mask]
                
                # Extract input and weight slices
                input_slice = input_patches[..., k_offsets_valid]  
                weight_slice = weight_patches[..., k_offsets_valid]
                
                # Accumulate the result
                accumulator[..., None] += input_slice * weight_slice[None, ...]
    
    # Apply GELU activation
    gelu_output = 0.5 * accumulator * (1.0 + tl.tanh(sqrt_2_over_pi * (accumulator + gelu_cubic_const * accumulator * accumulator * accumulator)))
    
    # Add bias
    final_output = gelu_output + bias_vector[:, None]
    
    # Store output
    output_ptrs = output_ptr + (m_offsets[:, None] * in_height * in_width + n_offsets[None, :])
    tl.store(output_ptrs, final_output, mask=(m_offsets[:, None] < out_channels) & (n_offsets[None, :] < in_height * in_width))

@torch.fx.wrap
def fused_conv3x3_gelu(input, weight, bias):
    batch_size, in_channels, in_height, in_width = input.shape
    out_channels, kernel_h, kernel_w = weight.shape[:3]
    groups = 2048  # Fixed for this pass
    
    # Output size for conv3x3 with padding=1, stride=1
    out_height = in_height
    out_width = in_width
    
    output = torch.empty((batch_size, out_channels, out_height, out_width), device=input.device, dtype=input.dtype)
    
    # Block sizes optimized for grouped convolution
    BLOCK_SIZE_M = 256   # Output channels per program (must be multiple of group size)
    BLOCK_SIZE_N = 512   # Spatial locations per program  
    BLOCK_SIZE_K = 16    # Input channels per loop
    
    # Grid calculation
    num_programs_m = (out_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_programs_n = (in_height * in_width + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel
    fused_conv3x3_gelu_kernel[(num_programs_m, num_programs_n)](
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
    return fused_conv3x3_gelu