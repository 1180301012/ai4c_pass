import torch
import triton
import triton.language as tl

def pattern(args):
    input, weight, bias = args
    conv = torch.conv2d(input, weight, bias, stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1024)
    gelu = torch.nn.functional.gelu(conv)
    return gelu

def replacement_args(args):
    return args

@triton.jit
def fused_conv_gelu_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_channels, in_height, in_width,
    out_channels, kernel_h, kernel_w,
    stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
    groups, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    # Calculate program indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Determine range of rows (output channels) each program handles
    m_mask = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) < out_channels
    
    # Create offsets for output
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # For groups=1024, we process each group independently
    group = m_offsets // (out_channels // groups)
    
    # Only compute within range
    m_offsets = m_offsets[m_mask]
    
    if len(m_offsets) == 0:
        return
    
    # Load bias vector for this group's output channels
    bias_vector = tl.load(bias_ptr + m_offsets, mask=m_mask)
    
    # Initialize accumulator
    accumulator = tl.zeros((len(m_offsets), BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over input channels per group
    channels_per_group = in_channels // groups
    k_offsets = tl.arange(0, channels_per_group)
    
    # For groups=1024 and 1x1 or 3x3 kernel, we optimize the memory access pattern
    for k in range(0, channels_per_group, BLOCK_SIZE_K):
        # Create masks for bounds checking
        k_mask = k + k_offsets < channels_per_group
        k_offsets_valid = k + k_offsets[k_mask]
        
        # Loop over kernel elements
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                # Calculate input offsets
                base_input_h = (pid_m * BLOCK_SIZE_M // groups) * stride_h + kh * dilation_h - pad_h
                base_input_w = (pid_n * BLOCK_SIZE_N // groups) * stride_w + kw * dilation_w - pad_w
                
                # Create coordinate matrices
                input_h_offsets = base_input_h + tl.arange(0, BLOCK_SIZE_N // groups * stride_h, stride_h)
                input_w_offsets = base_input_w + tl.arange(0, BLOCK_SIZE_N // groups * stride_w, stride_w)
                
                # Flatten coordinates for broadcasting
                input_h_offsets_flat = input_h_offsets[:, None].expand((input_h_offsets.shape[0], groups))
                input_w_offsets_flat = input_w_offsets[None, :].expand((input_h_offsets.shape[0], groups))
                
                # Load input patch
                input_ptrs = input_ptr + (input_h_offsets_flat * in_width * in_channels + input_w_offsets_flat * in_channels + group[:, None] * channels_per_group + k_offsets_valid[None, :])
                input_patch = tl.load(input_ptrs, mask=(input_h_offsets_flat >= 0) & (input_h_offsets_flat < in_height) & 
                                                   (input_w_offsets_flat >= 0) & (input_w_offsets_flat < in_width) &
                                                   (k_offsets_valid[None, :] >= 0) & (k_offsets_valid[None, :] < channels_per_group), other=0.0)
                
                # Load weight patch
                weight_ptrs = weight_ptr + (m_offsets[:, None] * kernel_h * kernel_w * channels_per_group + 
                                           kh * kernel_w * channels_per_group + kw * channels_per_group + 
                                           k_offsets_valid[None, :])
                weight_patch = tl.load(weight_ptrs, mask=k_offsets_valid[None, :] >= 0, other=0.0)
                
                # Matrix multiply
                accumulator += input_patch * weight_patch[None, :]
    
    # Apply GELU activation
    # Using GELU approximation: 0.5 * x * (1.0 + tanh(sqrt(2.0 / pi) * (x + 0.044715 * x^3)))
    output_accumulator = 0.5 * accumulator * (1.0 + tl.tanh(tl.sqrt(2.0 / tl.pi) * (accumulator + 0.044715 * accumulator * accumulator * accumulator)))
    
    # Add bias
    output_final = output_accumulator + bias_vector[:, None]
    
    # Store output
    output_ptrs = output_ptr + (m_offsets[:, None] * in_height * in_width + 
                               tl.arange(0, BLOCK_SIZE_N)[:, None].expand((len(m_offsets), BLOCK_SIZE_N)))
    
    valid_mask = m_offsets[:, None] < out_channels
    tl.store(output_ptrs, output_final, mask=valid_mask)

@torch.fx.wrap
def fused_conv_gelu(input, weight, bias):
    # Get tensor shapes
    batch_size, in_channels, in_height, in_width = input.shape
    out_channels, kernel_h, kernel_w = weight.shape[:3]
    groups = 1024  # Fixed for this pass
    
    # Determine output size
    out_height = ((in_height + 2 * 1 - kernel_h) // 1) + 1  # padding=1, stride=1
    out_width = ((in_width + 2 * 1 - kernel_w) // 1) + 1
    
    # Create output tensor
    output = torch.empty((batch_size, out_channels, out_height, out_width), device=input.device, dtype=input.dtype)
    
    # Block sizes for GPU optimization
    BLOCK_SIZE_M = 128  # Output channels per program
    BLOCK_SIZE_N = 64   # Spatial locations per program
    BLOCK_SIZE_K = 32   # Input channels per loop
    
    # Calculate grid size
    num_programs_m = (out_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_programs_n = (out_height * out_width + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel
    fused_conv_gelu_kernel[(num_programs_m, num_programs_n)](
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
    return fused_conv_gelu