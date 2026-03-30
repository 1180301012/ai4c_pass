import torch
import triton
import triton.language as tl

# Pattern matching function - matches Conv2D + View + Softmax sequence for float32/0 pattern
def pattern(in_0, in_1, in_2):
    """Match Conv2D + View + Softmax pattern for batch size 1"""
    # Conv2D operation with exact parameters from the model (positional args only)
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    
    # View operation to reshape for softmax - EXACT pattern from float32/0
    tmp_3 = tmp_2.view(1, 1, -1)
    
    # Softmax along the last dimension
    tmp_4 = tmp_3.softmax(dim=-1)
    
    # Return what the original model returns
    return (tmp_4,)

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    """Extract arguments for the replacement function"""
    return (in_0, in_1, in_2)

# Optimized Triton kernel that fuses Conv2D → View → Softmax
@triton.jit
def fused_conv_view_softmax_kernel(
    # Input tensors
    in_2_ptr,  # input feature map [B, C_in, H, W]
    in_1_ptr,  # weights [C_out, C_in/groups, KH, KW] 
    in_0_ptr,  # bias [C_out]
    # Output tensor  
    out_ptr,   # output [B, 1, H*W*C_out]
    # Parameters
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    # Conv2D parameters (fixed from model)
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    groups: tl.constexpr,
    # Data type
    BLOCK_SIZE: tl.constexpr,
):
    """Fused Conv2D + Reshape + Softmax kernel"""
    
    # Calculate total elements per output element (H*W*C_out)
    spatial_elements = height * width * out_channels
    
    # Each program handles one spatial element across the entire batch
    program_id = tl.program_id(0)
    
    # Output coordinates for softmax: [batch_idx, 0, spatial_idx]
    batch_idx = program_id // spatial_elements
    spatial_idx = program_id % spatial_elements
    
    if batch_idx >= batch_size:
        return
        
    # Convert spatial_idx back to conv2d coordinates
    out_c = spatial_idx // (height * width)
    spatial_flat_idx = spatial_idx % (height * width)
    h_idx = spatial_flat_idx // width
    w_idx = spatial_flat_idx % width
    
    # Calculate input position for conv2d
    oh = h_idx * stride_h - pad_h
    ow = w_idx * stride_w - pad_w
    
    # Initialize accumulation
    accumulator = 0.0
    
    # Perform convolution
    for ci in range(in_channels):
        # Handle weight dimensions [C_out, C_in/groups, KH, KW]
        weight_ptr_base = in_1_ptr + out_c * (in_channels // groups) * 1 * 1 + ci * 1 * 1
        
        # Since KH=1, KW=1, we only need center weight
        weight_val = tl.load(weight_ptr_base + 0 * 1 * 1 + 0 * 1 + 0, mask=None)
        
        # Calculate input position with bounds checking
        ih = oh + dilation_h * 0
        iw = ow + dilation_w * 0
        
        ih_valid = (ih >= 0) & (ih < height)
        iw_valid = (iw >= 0) & (iw < width)
        
        if ih_valid and iw_valid:
            input_ptr_val = in_2_ptr + batch_idx * in_channels * height * width + ci * height * width + ih * width + iw
            input_val = tl.load(input_ptr_val, mask=None)
            accumulator += weight_val * input_val
    
    # Add bias (using output channel index)
    bias_val = tl.load(in_0_ptr + out_c, mask=None)
    accumulator += bias_val
    
    # Store result (this is before softmax)
    tl.store(out_ptr + batch_idx * spatial_elements + spatial_idx, accumulator, mask=None)

# Kernel wrapper that handles grid setup and softmax
@torch.fx.wrap
def fused_conv_view_softmax(in_0, in_1, in_2):
    """Conv2D + Reshape + Softmax fusion kernel wrapper"""
    
    # Get tensor shapes
    batch_size, in_channels, height, width = in_2.shape
    out_channels = in_1.shape[0]
    
    # Calculate output shape for softmax: [batch_size, 1, height*width*out_channels]
    spatial_elements = height * width * out_channels
    total_elements = batch_size * spatial_elements
    
    # Output tensor
    out = torch.empty((batch_size, 1, spatial_elements), dtype=in_2.dtype, device=in_2.device)
    
    # Optimize block size based on data type
    if in_2.dtype == torch.float32:
        BLOCK_SIZE = 1024
    elif in_2.dtype in [torch.float16, torch.bfloat16]:
        BLOCK_SIZE = 2048
    else:
        BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel for Conv2D part
    fused_conv_view_softmax_kernel[(grid_size,)](
        in_2_ptr=in_2,
        in_1_ptr=in_1,
        in_0_ptr=in_0,
        out_ptr=out,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        height=height,
        width=width,
        stride_h=1,
        stride_w=1,
        pad_h=0,
        pad_w=0,
        dilation_h=1,
        dilation_w=1,
        groups=1,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Optimized 1D softmax kernel (decorated)
@triton.jit
def softmax_1d_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Optimized 1D softmax kernel"""
    program_id = tl.program_id(0)
    start_idx = program_id * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load batch of data
    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
    
    # Compute max for numerical stability
    max_val = tl.max(x)
    
    # Subtract max and compute exp
    x = x - max_val
    exp_x = tl.exp(x)
    
    # Compute sum
    sum_exp = tl.sum(exp_x)
    
    # Normalize
    softmax = exp_x / sum_exp
    tl.store(out_ptr + offsets, softmax, mask=mask)

# Apply softmax to each batch
def fused_conv_view_softmax(in_0, in_1, in_2):
    """Conv2D + Reshape + Softmax fusion kernel wrapper"""
    
    # Get tensor shapes
    batch_size, in_channels, height, width = in_2.shape
    out_channels = in_1.shape[0]
    
    # Calculate output shape for softmax: [batch_size, 1, height*width*out_channels]
    spatial_elements = height * width * out_channels
    total_elements = batch_size * spatial_elements
    
    # Output tensor
    out = torch.empty((batch_size, 1, spatial_elements), dtype=in_2.dtype, device=in_2.device)
    
    # Optimize block size based on data type
    if in_2.dtype == torch.float32:
        BLOCK_SIZE = 1024
    elif in_2.dtype in [torch.float16, torch.bfloat16]:
        BLOCK_SIZE = 2048
    else:
        BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel for Conv2D part
    fused_conv_view_softmax_kernel[(grid_size,)](
        in_2_ptr=in_2,
        in_1_ptr=in_1,
        in_0_ptr=in_0,
        out_ptr=out,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        height=height,
        width=width,
        stride_h=1,
        stride_w=1,
        pad_h=0,
        pad_w=0,
        dilation_h=1,
        dilation_w=1,
        groups=1,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Apply softmax to each batch
    for batch_idx in range(batch_size):
        batch_data = out[batch_idx, 0, :]  # Shape: [H*W*C_out]
        softmax_out = torch.empty_like(batch_data)
        
        n_elements = batch_data.numel()
        grid_size_softmax = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        softmax_1d_kernel[(grid_size_softmax,)](
            batch_data, softmax_out, n_elements, BLOCK_SIZE
        )
        
        out[batch_idx, 0, :] = softmax_out
    
    return out

# Replacement function (no arguments, returns function reference)
def replacement_func():
    return fused_conv_view_softmax