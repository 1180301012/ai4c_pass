import torch
import triton
import triton.language as tl

def pattern(x, weight, bias):
    tmp_2 = torch.conv2d(x, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = tmp_2.sigmoid()
    return tmp_3

def replacement_args(x, weight, bias):
    return (x, weight, bias)

@triton.jit
def conv2d_sigmoid_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    batch_size,
    in_channels,
    out_channels,
    input_height,
    input_width,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Each program handles BLOCK_SIZE_M output channels and BLOCK_SIZE_N spatial positions
    pid = tl.program_id(0)
    
    # Divide work: pid=0: batch, pid=1: output channels, pid=2: spatial positions  
    batch_idx = pid // ((out_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M * 
                      ((input_height * input_width) + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)
    pid_remaining = pid % ((out_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M * 
                           ((input_height * input_width) + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)
    
    out_channel_idx = pid_remaining // ((input_height * input_width) + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    spatial_idx = pid_remaining % ((input_height * input_width) + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Calculate actual channel and spatial ranges
    out_start = out_channel_idx * BLOCK_SIZE_M
    out_end = min(out_start + BLOCK_SIZE_M, out_channels)
    spatial_start = spatial_idx * BLOCK_SIZE_N
    spatial_end = min(spatial_start + BLOCK_SIZE_N, input_height * input_width)
    
    # Ensure we don't go out of bounds
    if batch_idx >= batch_size:
        return
    
    # Initialize output tile  
    for oc in range(out_start, out_end):
        bias_val = tl.load(bias_ptr + oc)
        output_base = (batch_idx * out_channels + oc) * input_height * input_width
        for spatial_pos in range(spatial_start, spatial_end):
            if spatial_pos < input_height * input_width and oc < out_channels:
                out_addr = out_ptr + output_base + spatial_pos
                tl.store(out_addr, bias_val)
    
    # Compute matrix multiplication: output = bias + input * weight^T
    # For 1x1 conv, weights[oc, ic] computes contribution from input_channel ic to output_channel oc
    for oc in range(out_start, out_end):
        # Process only if in bounds for output channels
        if oc < out_channels:
            for ic in range(in_channels):
                # Load weight for this output and input channel
                weight_val = tl.load(weight_ptr + oc * in_channels + ic)
                
                # Compute contribution to all spatial positions
                output_base = (batch_idx * out_channels + oc) * input_height * input_width
                input_base = (batch_idx * in_channels + ic) * input_height * input_width
                
                for spatial_pos in range(spatial_start, spatial_end):
                    if spatial_pos < input_height * input_width:
                        in_addr = x_ptr + input_base + spatial_pos
                        out_addr = out_ptr + output_base + spatial_pos
                        
                        # Load existing output value and add contribution
                        existing_output = tl.load(out_addr)
                        input_val = tl.load(in_addr)
                        new_val = existing_output + weight_val * input_val
                        
                        tl.store(out_addr, new_val)
    
    # Apply sigmoid activation
    for oc in range(out_start, out_end):
        # Apply sigmoid only if in bounds for output channels
        if oc < out_channels:
            output_base = (batch_idx * out_channels + oc) * input_height * input_width
            for spatial_pos in range(spatial_start, spatial_end):
                if spatial_pos < input_height * input_width:
                    out_addr = out_ptr + output_base + spatial_pos
                    val = tl.load(out_addr)
                    tl.store(out_addr, tl.sigmoid(val))

@torch.fx.wrap
def conv2d_sigmoid_impl(x, weight, bias):
    batch_size, in_channels, input_height, input_width = x.shape
    out_channels, _, _, _ = weight.shape
    
    # Block sizes for matrix multiplication pattern
    BLOCK_SIZE_M = 32  # Output channels per block
    BLOCK_SIZE_N = 256  # Spatial positions per block
    
    # Grid size calculation using simpler 1D grid
    total_spatial_positions = input_height * input_width
    
    grid_dim_x = (out_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_dim_y = (total_spatial_positions + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_dim_z = batch_size
    
    total_programs = grid_dim_x * grid_dim_y * grid_dim_z
    grid = (total_programs,)
    
    out = torch.empty((batch_size, out_channels, input_height, input_width), dtype=torch.float32, device=x.device)
    
    if total_programs > 0:
        conv2d_sigmoid_kernel[grid](
            x_ptr=x,
            weight_ptr=weight,
            bias_ptr=bias,
            out_ptr=out,
            batch_size=batch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            input_height=input_height,
            input_width=input_width,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )
    
    return out

def replacement_func():
    return conv2d_sigmoid_impl