import torch
import triton
import triton.language as tl

# Pattern matching function - must mirror the exact operations in model.py
def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 4)
    tmp_3 = torch.sigmoid(conv2d)
    tmp_4 = tmp_3.view(1, -1, 1, 1)
    tmp_5 = in_2 * tmp_4
    tmp_6 = tmp_5.contiguous()
    return tmp_6

# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_conv_sigmoid_mul_kernel(
    x_ptr,           # in_3: input [1, C_in, 1, 1]
    weight_ptr,      # in_1: weights [C_out, C_in_per_group, 1, 1]  
    bias_ptr,        # in_0: bias [C_out]
    y_ptr,           # in_2: feature map [1, C_out, H, W]
    out_ptr,         # output [1, C_out, H, W]
    batch_size,      # always 1
    in_channels,     # C_in = 32
    out_channels,    # C_out = 96
    height,          # H (varies: 112, 128, 160, etc.)
    width,           # W (varies: 112, 128, 160, etc.)
    groups,          # groups = 4
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Each program handles one spatial location (H, W)
    h = tl.program_id(0)
    w = tl.program_id(1)
    c_out = tl.program_id(2)
    
    # Convolution result for this specific output channel
    conv_val = 0.0
    
    # Find which group this output channel belongs to and process grouped convolution
    in_channels_per_group = in_channels // groups
    out_channels_per_group = out_channels // groups
    
    # Directly calculate the group for this output channel
    group_id = c_out // out_channels_per_group
    g_in_start = group_id * in_channels_per_group
    g_out_start = group_id * out_channels_per_group
    local_c_out = c_out - g_out_start
    
    # Load bias for this output channel
    bias = tl.load(bias_ptr + c_out, eviction_policy='evict_last')
    
    # Load weight for this output channel in this group
    # Note: Triton expects [C_out, C_in_per_group, 1, 1] layout
    weight = tl.load(weight_ptr + (c_out * in_channels_per_group), eviction_policy='evict_last')
    
    # For grouped conv2d, sum over input channels in this group
    for c_in_in_group in range(in_channels_per_group):
        in_idx = g_in_start + c_in_in_group
        if in_idx < in_channels:
            # Load input value
            x_val = tl.load(x_ptr + in_idx, eviction_policy='evict_last')
            # Add to convolution result
            conv_val += weight * x_val
    
    # Add bias (each output channel gets only one bias)
    conv_val += bias
    
    # Apply sigmoid to convolution result
    sigmoid_val = 1.0 / (1.0 + tl.exp(-conv_val))
    
    # Load and multiply with feature map
    y_offset = c_out * height * width + h * width + w
    y_val = tl.load(y_ptr + y_offset, eviction_policy='evict_last')
    
    # Broadcast and multiply: sigmoid_val * y_val
    result = sigmoid_val * y_val
    
    # Store result
    out_offset = c_out * height * width + h * width + w
    tl.store(out_ptr + out_offset, result)

@triton.jit
def fused_conv_sigmoid_mul_kernel_simple(
    x_ptr,           # in_3: input [1, C_in, 1, 1]
    weight_ptr,      # in_1: weights [C_out, C_in_per_group, 1, 1]  
    bias_ptr,        # in_0: bias [C_out]
    y_ptr,           # in_2: feature map [1, C_out, H, W]
    out_ptr,         # output [1, C_out, H, W]
    batch_size,      # always 1
    in_channels,     # C_in = 32
    out_channels,    # C_out = 96
    height,          # H (varies: 112, 128, 160, etc.)
    width,           # W (varies: 112, 128, 160, etc.)
    groups,          # groups = 4
    SPATIAL_TILE: tl.constexpr,
):
    # Each program processes one channel and part of spatial dimensions
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_c = tl.program_id(2)
    
    # Process spatial in tiles for better efficiency
    h_start = pid_h * SPATIAL_TILE
    h_end = min(h_start + SPATIAL_TILE, height)
    w_start = pid_w * SPATIAL_TILE
    w_end = min(w_start + SPATIAL_TILE, width)
    
    # Process this output channel
    local_c = pid_c
    
    # For grouped conv2d, find the right group and process it
    in_channels_per_group = in_channels // groups
    out_channels_per_group = out_channels // groups
    group_id = local_c // out_channels_per_group
    local_group_c = local_c % out_channels_per_group
    
    # Load bias for this output channel
    bias = tl.load(bias_ptr + local_c, eviction_policy='evict_last')
    
    # Simple 1x1 grouped convolution: sum over input channels in this group
    conv_val = 0.0
    for c_in in range(in_channels_per_group):
        in_idx = group_id * in_channels_per_group + c_in
        if in_idx < in_channels:
            x_val = tl.load(x_ptr + in_idx, eviction_policy='evict_last')
            # Weight for this input and output channel in the group
            weight = tl.load(weight_ptr + (local_c * in_channels_per_group + c_in), eviction_policy='evict_last')
            conv_val += weight * x_val
    
    # Add bias and apply sigmoid
    conv_val += bias
    sigmoid_val = 1.0 / (1.0 + tl.exp(-conv_val))
    
    # Process all spatial locations in this tile
    for h in range(h_start, h_end):
        for w in range(w_start, w_end):
            y_offset = local_c * height * width + h * width + w
            y_val = tl.load(y_ptr + y_offset, eviction_policy='evict_last')
            result = sigmoid_val * y_val
            out_offset = local_c * height * width + h * width + w
            tl.store(out_ptr + out_offset, result)

@torch.fx.wrap
def fused_conv_sigmoid_mul(in_0, in_1, in_2, in_3):
    # Get tensor shapes and metadata
    batch_size, in_channels, input_h, input_w = in_3.shape
    _, out_channels, kernel_h, kernel_w = in_1.shape
    _, _, feature_h, feature_w = in_2.shape
    groups = 4
    
    # Create output tensor
    out = torch.empty_like(in_2)
    
    # Use moderate spatial tile size for good balance of efficiency and overhead
    SPATIAL_TILE = 32  # Process 32x32 spatial blocks per program
    
    # Calculate grid dimensions - each program handles one channel and a spatial tile
    num_h_tiles = (feature_h + SPATIAL_TILE - 1) // SPATIAL_TILE
    num_w_tiles = (feature_w + SPATIAL_TILE - 1) // SPATIAL_TILE
    
    grid = (
        num_h_tiles,    # spatial tiles in height
        num_w_tiles,    # spatial tiles in width
        out_channels    # one program per output channel
    )
    
    # Use the simple kernel that performed best in our tests
    fused_conv_sigmoid_mul_kernel_simple[grid](
        x_ptr=in_3,
        weight_ptr=in_1,
        bias_ptr=in_0,
        y_ptr=in_2,
        out_ptr=out,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        height=feature_h,
        width=feature_w,
        groups=groups,
        SPATIAL_TILE=SPATIAL_TILE,
    )
    
    return out

# Replacement function (returns function reference, not a call)
def replacement_func():
    return fused_conv_sigmoid_mul