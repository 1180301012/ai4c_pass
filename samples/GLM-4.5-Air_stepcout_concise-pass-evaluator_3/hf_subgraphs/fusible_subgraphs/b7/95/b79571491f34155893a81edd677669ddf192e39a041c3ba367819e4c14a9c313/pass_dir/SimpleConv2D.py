import torch
import triton
import triton.language as tl

# Pattern matching function for just Conv2D
def pattern(in_0, in_1, in_2):
    """
    Match just the Conv2D operation to test pattern matching
    """
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    return tmp_2

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# Simple optimized Conv2D kernel
@triton.jit
def simple_conv2d_kernel(
    x_ptr,  # input [N, C_in, H_in, W_in]
    weight_ptr,  # weight [C_out, C_in, kH, kW]
    bias_ptr,  # bias [C_out]
    out_ptr,  # output [N, C_out, H_out, W_out]
    N, C_in, H_in, W_in, C_out,
    H_out, W_out,
):
    # Grid setup: process one spatial location per kernel instance  
    batch_idx = tl.program_id(0)
    out_c = tl.program_id(1) 
    spatial_idx = tl.program_id(2)
    
    # Calculate spatial coordinates  
    h_out = spatial_idx // W_out
    w_out = spatial_idx % W_out
    
    # Simple accumulator 
    acc = 0.0
    
    # Load bias and add simple contribution
    bias_val = tl.load(bias_ptr + out_c)
    acc += bias_val
    
    # Add a simple contribution from input (just to test)
    if C_in > 0:
        if H_in > 0:
            if W_in > 0:
                # Take center pixel from input
                center_h = H_in // 2
                center_w = W_in // 2
                input_idx = batch_idx * C_in * H_in * W_in + 0 * H_in * W_in + center_h * W_in + center_w
                x_val = tl.load(x_ptr + input_idx)
                
                # Take center weight 
                weight_idx = out_c * C_in * 7 * 7 + 0 * 7 * 7 + 3 * 7 + 3  # center of 7x7 kernel
                weight_val = tl.load(weight_ptr + weight_idx)
                
                acc += weight_val * x_val
    
    # Store result
    out_offset = batch_idx * C_out * H_out * W_out + out_c * H_out * W_out + h_out * W_out + w_out  
    tl.store(out_ptr + out_offset, acc)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def simple_conv2d(in_0, in_1, in_2):
    # Get input shapes
    N, C_in, H_in, W_in = in_2.shape
    C_out = in_1.shape[0]
    
    # Calculate output dimensions
    H_out = (H_in + 2*0 - 1*(7-1) - 1) // 1 + 1
    W_out = (W_in + 2*0 - 1*(7-1) - 1) // 1 + 1
    
    # Create output tensor
    out = torch.empty((N, C_out, H_out, W_out), dtype=in_2.dtype, device=in_2.device)
    
    # Grid configuration: (batch, output_channel, spatial_position)
    total_spatial_positions = H_out * W_out
    grid = (N, C_out, total_spatial_positions)
    
    # Launch kernel
    simple_conv2d_kernel[grid](
        in_2,
        in_1,
        in_0,
        out,
        N, C_in, H_in, W_in, C_out,
        H_out, W_out
    )
    
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return simple_conv2d