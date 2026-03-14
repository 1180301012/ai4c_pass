import torch
import triton
import triton.language as tl

# Pattern matching function - match both Conv2D AND HardTanh together
def pattern(in_0, in_1, in_2, in_3):
    """
    Match both operations:
    tmp_0 = in_0
    tmp_1 = in_1  
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_1 = tmp_0 = None
    tmp_3 = torch.nn.functional.hardtanh(in_3, 0.0, 6.0, False)
    return (tmp_3, tmp_2)
    """
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    return (torch.nn.functional.hardtanh(in_3, 0.0, 6.0, False), tmp_2)

# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

# Optimized HardTanh kernel using Triton
@triton.jit
def hardtanh_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    min_val: tl.constexpr,
    max_val: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds
    
    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply HardTanh: y = min(max(x, min_val), max_val)
    y = tl.where(x < min_val, min_val, x)
    y = tl.where(y > max_val, max_val, y)
    
    # Store result
    tl.store(output_ptr + offsets, y, mask=mask)

# Optimized 1x1 Conv2D kernel with blocking for efficiency
@triton.jit
def conv2d_1x1_kernel(
    input_ptr,      # [B, C_in, H, W] 
    weight_ptr,     # [C_out, C_in, 1, 1]
    bias_ptr,       # [C_out]
    output_ptr,     # [B, C_out, H, W]
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    # Get program IDs - 2D grid for efficiency
    pid_c = tl.program_id(0)  # Output channels dimension
    pid_hw = tl.program_id(1)  # Combined spatial dimension
    
    # Compute ranges for output channels
    c_offset = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    c_mask = c_offset < out_channels
    
    # Compute spatial indices from combined index
    spatial_offset = pid_hw * BLOCK_SIZE_HW + tl.arange(0, BLOCK_SIZE_HW)
    hw_mask = spatial_offset < (height * width)
    
    # Convert spatial offset to (h, w) coordinates
    h_offset = spatial_offset // width
    w_offset = spatial_offset % width
    
    # Load bias for current output channels
    bias_vals = tl.load(bias_ptr + c_offset, mask=c_mask, other=0.0)
    
    # Initialize accumulators for all channels in block
    acc = tl.zeros((BLOCK_SIZE_HW, BLOCK_SIZE_C), dtype=tl.float32)
    
    # Process input channels for all positions in block
    for c_in in range(in_channels):
        # Load weight for current output channels and input channel
        weight_vals = tl.load(weight_ptr + c_offset[:, None] * in_channels + c_in,
                            mask=(c_mask[:, None] & True), other=0.0)
        
        # Load input for all spatial positions in batch
        input_base = c_in * height * width
        for i_hw in range(BLOCK_SIZE_HW):
            if hw_mask[i_hw]:
                input_idx = input_base + spatial_offset[i_hw]
                input_val = tl.load(input_ptr + input_idx, mask=hw_mask[i_hw], other=0.0)
                
                # Accumulate for all output channels
                for i_c in range(BLOCK_SIZE_C):
                    if c_mask[i_c]:
                        acc[i_hw, i_c] += input_val * weight_vals[i_c, 0]
    
    # Add bias
    for i_hw in range(BLOCK_SIZE_HW):
        if hw_mask[i_hw]:
            for i_c in range(BLOCK_SIZE_C):
                if c_mask[i_c]:
                    acc[i_hw, i_c] += bias_vals[i_c]
    
    # Store output for all channels and positions
    for i_hw in range(BLOCK_SIZE_HW):
        if hw_mask[i_hw]:
            # Compute output spatial position
            h_idx = h_offset[i_hw]
            w_idx = w_offset[i_hw]
            
            # Store all output channels for this spatial position
            for batch_idx in range(batch_size):
                output_base = batch_idx * out_channels * height * width + h_idx * width + w_idx
                for i_c in range(BLOCK_SIZE_C):
                    if c_mask[i_c]:
                        output_idx = output_base + i_c * height * width
                        tl.store(output_ptr + output_idx, acc[i_hw, i_c])

# Optimized HardTanh wrapper
@torch.fx.wrap
def optimized_hardtanh(input, min_val=0.0, max_val=6.0):
    # Get total number of elements
    n_elements = input.numel()
    
    # Use optimal block size for GPU
    BLOCK_SIZE = 1024
    
    # Calculate number of programs needed
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Prepare output tensor
    output = torch.empty_like(input)
    
    # Launch kernel
    hardtanh_kernel[(num_programs,)](
        input_ptr=input,
        output_ptr=output,
        n_elements=n_elements,
        min_val=min_val,
        max_val=max_val,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Optimized Conv2D + HardTanh fusion wrapper
@torch.fx.wrap  
def fused_optimized_operations(bias, weight, input_tensor,hardtanh_input):
    # Separate tensors
    input_conv, input_hardtanh = input_tensor, hardtanh_input
    
    # Validate input tensor shapes
    if input_conv.dim() != 4:
        raise ValueError(f"Expected 4D input tensor [B, C_in, H, W], got {input_conv.dim()}D: {input_conv.shape}")
    if weight.dim() != 4:
        raise ValueError(f"Expected 4D weight tensor [C_out, C_in, 1, 1], got {weight.dim()}D: {weight.shape}")
    
    # Extract tensor dimensions
    batch_size, in_channels, height, width = input_conv.shape
    out_channels = weight.shape[0]
    
    # Handle bias - ensure [C_out]
    if bias.dim() == 1:
        pass  # expected
    elif bias.dim() == 0:
        bias = bias.unsqueeze(0)  # Make it 1D
    else:
        raise ValueError(f"Unsupported bias dimension: {bias.dim()}, shape: {bias.shape}")
    
    # Optimize HardTanh first (faster, independent operation)  
    hardtanh_output = optimized_hardtanh(input_hardtanh)
    
    # Calculate efficient block sizes for Conv2D
    BLOCK_SIZE_C = min(32, out_channels)  # Output channels per program
    BLOCK_SIZE_HW = min(1024, height * width)  # Spatial elements per program
    
    # Calculate grid size
    grid_c = (out_channels + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    grid_hw = (height * width + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW
    
    # Prepare output tensor: [B, C_out, H, W]
    conv_output = torch.empty((batch_size, out_channels, height, width), 
                             dtype=input_conv.dtype, device=input_conv.device)
    
    # Launch Conv2D kernel
    conv2d_1x1_kernel[(grid_c, grid_hw)](
        input_ptr=input_conv,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=conv_output,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        height=height,
        width=width,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_HW=BLOCK_SIZE_HW,
    )
    
    return (hardtanh_output, conv_output)

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_optimized_operations