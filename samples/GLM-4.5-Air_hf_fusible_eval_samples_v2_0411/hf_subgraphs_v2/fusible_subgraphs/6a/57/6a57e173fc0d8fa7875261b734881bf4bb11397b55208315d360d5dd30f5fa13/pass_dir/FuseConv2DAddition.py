import torch

# Pattern matching function - exactly like the reference
def pattern(x, y):
    return torch.add(x, y)

# Argument extraction function - exactly like the reference  
def replacement_args(x, y):
    return (x, y)

# Placeholder kernel - using the reference example structure
@triton.jit
def triton_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_add(x, y):
    import triton.language as tl
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty_like(x)
    triton_add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

# Replacement function - exactly like the reference
def replacement_func():
    return triton_add
    input_ptr,
    stride_h, stride_w,
    pad_h, pad_w,
    dilation_h, dilation_w,
    groups,
    batch_size, in_channels, in_height, in_width, out_channels, kernel_height, kernel_width,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused Conv2D + Addition kernel
    Performs group convolution and adds result to output tensor in-place
    """
    # Get program ID for work distribution
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_g = tl.program_id(2)  # Group ID
    
    # Calculate range for this program
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    g_start = pid_g * (out_channels // groups)
    g_end = (pid_g + 1) * (out_channels // groups)
    
    # Create offsets within the block
    offsets_m = m_start + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = n_start + tl.arange(0, BLOCK_SIZE_N)
    offsets_g = tl.arange(g_start, g_end)
    
    # Initialize output accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Handle group convolution
    local_k_offset = tl.arange(0, BLOCK_SIZE_K)
    
    for k in range(0, in_channels // groups, BLOCK_SIZE_K):
        k_end = min(k + BLOCK_SIZE_K, in_channels // groups)
        
        # Load weights for this group and block
        weight_offsets = (
            offsets_g[:, None] * (in_channels // groups * kernel_height * kernel_width) +
            pid_g * (in_channels // groups) * kernel_height * kernel_width +
            local_k_offset[None, :] * kernel_height * kernel_width
        )
        weight = tl.load(weight_ptr + weight_offsets, mask=(tl.arange(k_end - k) < (k_end - k))[None, :], other=0.0)
        
        # Load input data
        input_offsets = (
            offsets_m[:, None] * (in_channels // groups * in_height * in_width) +
            pid_g * (in_channels // groups) * in_height * in_width +
            local_k_offset[None, :] * in_height * in_width
        )
        input_val = tl.load(input_ptr + input_offsets, mask=(tl.arange(k_end - k) < (k_end - k))[None, :], other=0.0)
        
        # Compute GEMM operation for this block
        accumulator += tl.dot(input_val, weight.to(tl.float32))
    
    # Load existing output data for addition
    output_offsets = (
        offsets_m[:, None] * out_channels +
        offsets_n[None, :]
    )
    existing_output = tl.load(out_ptr + output_offsets, mask=(
        (offsets_m[:, None] < batch_size) & (offsets_n[None, :] < out_channels // batch_size)
    ), other=0.0)
    
    # Add convolution result to existing output
    result = accumulator + existing_output
    
    # Store result back
    tl.store(out_ptr + output_offsets, result, mask=(
        (offsets_m[:, None] < batch_size) & (offsets_n[None, :] < out_channels // batch_size)
    ))

@torch.fx.wrap
def fused_conv2d_add(weight, input_tensor, output_tensor):
    """
    Wrapper function for fused Conv2D + Addition
    """
    # Get tensor shapes
    batch_size, in_channels, in_height, in_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Convolution parameters
    stride_h, stride_w = 1, 1
    pad_h, pad_w = 32, 0
    dilation_h, dilation_w = 1, 1
    groups = 4
    
    # Calculate output dimensions
    out_height = (in_height + 2 * pad_h - dilation_h * (kernel_height - 1) - 1) // stride_h + 1
    out_width = (in_width + 2 * pad_w - dilation_w * (kernel_width - 1) - 1) // stride_w + 1
    
    # Set block sizes for Triton
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    # Calculate grid dimensions
    num_blocks_m = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_blocks_n = (out_channels + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_groups = groups
    
    # Launch kernel
    fused_conv2d_add_kernel[(num_blocks_m, num_blocks_n, num_groups)](
        output_tensor,
        weight,
        input_tensor,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w,
        groups,
        batch_size, in_channels, in_height, in_width,
        out_channels, kernel_height, kernel_width,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
    )
    
    return output_tensor

# Replacement function (returns function reference)
def replacement_func():
    return fused_conv2d_add