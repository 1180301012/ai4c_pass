import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    """Pattern matching the conv3d operation"""
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = in_2
    tmp_3 = torch.conv3d(in_3, tmp_1, tmp_0, (2, 16, 16), (0, 0, 0), (1, 1, 1), 1)
    return tmp_3

def replacement_args(in_0, in_1, in_2, in_3):
    """Extract arguments for the fused kernel"""
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_conv3d_flatten_transpose_kernel(
    input_ptr, weight_ptr, bias_ptr,
    output_ptr,
    batch_size, in_channels, out_channels,
    input_depth, input_height, input_width,
    weight_depth, weight_height, weight_width,
    output_depth, output_height, output_width,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel: conv3d + flatten + transpose"""
    
    # Get program ID for parallel execution
    pid = tl.program_id(0)
    
    # Each program handles BLOCK_SIZE output positions in the flattened space
    output_flat_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total_output_positions = output_depth * output_height * output_width
    output_mask = output_flat_idx < total_output_positions
    
    if not output_mask.any():
        return
    
    # Convert flat index back to 3D spatial coordinates for output
    # We want to map: [flat_idx] -> [d, h, w] where d is fastest changing
    output_idx_w = output_flat_idx // (output_depth * output_height)  # slowest changing
    remainder = output_flat_idx % (output_depth * output_height)
    output_idx_h = remainder // output_depth  # medium
    output_idx_d = remainder % output_depth  # fastest changing
    
    # Compute corresponding input indices for convolution
    # Output (d, h, w) maps to input (2*d + kd, 16*h + kh, 16*w + kw)
    input_base_idx_d = output_idx_d * 2
    input_base_idx_h = output_idx_h * 16  
    input_base_idx_w = output_idx_w * 16
    
    # Accumulate for each output channel
    for oc in range(out_channels):
        # Initialize accumulator
        accum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        
        # Iterate over input channels and kernel spatial dimensions
        for ic in range(in_channels):
            for kd in range(weight_depth):
                for kh in range(weight_height):
                    for kw in range(weight_width):
                        # Compute absolute input coordinates
                        abs_input_d = input_base_idx_d + kd
                        abs_input_h = input_base_idx_h + kh
                        abs_input_w = input_base_idx_w + kw
                        
                        # Check bounds
                        if (abs_input_d < input_depth and abs_input_h < input_height and abs_input_w < input_width):
                            # Calculate input tensor offset: [B, C, D, H, W]
                            input_offset = (ic * input_depth * input_height * input_width +
                                          abs_input_d * input_height * input_width +
                                          abs_input_h * input_width +
                                          abs_input_w)
                            
                            # Calculate weight tensor offset: [C_out, C_in, K_d, K_h, K_w]
                            weight_offset = (oc * in_channels * weight_depth * weight_height * weight_width +
                                           ic * weight_depth * weight_height * weight_width +
                                           kd * weight_height * weight_width +
                                           kh * weight_width +
                                           kw)
                            
                            # Load data
                            input_val = tl.load(input_ptr + input_offset, mask=output_mask, other=0.0)
                            weight_val = tl.load(weight_ptr + weight_offset, mask=output_mask, other=0.0)
                            
                            # Multiply and accumulate
                            accum += input_val * weight_val
        
        # Add bias
        bias_val = tl.load(bias_ptr + oc, mask=output_mask, other=0.0)
        output_val = accum + bias_val
        
        # Store in transposed format: [B, output_flat_idx, C_out]
        # Output is arranged as [batch, spatial_positions, channels]
        output_offset = (output_flat_idx * out_channels + oc)
        tl.store(output_ptr + output_offset, output_val, mask=output_mask)

@torch.fx.wrap
def fused_conv3d_flatten_transpose(in_0, in_1, in_2, in_3):
    """Wrapper for the fused conv3d + flatten + transpose kernel"""
    
    # Get input tensor shapes
    batch_size, in_channels, input_depth, input_height, input_width = in_3.shape
    in_channels_, out_channels, weight_depth, weight_height, weight_width = in_1.shape
    assert in_channels == in_channels_, "Input channels mismatch"
    
    # Calculate output shapes
    output_depth = input_depth // 2
    output_height = input_height // 16  
    output_width = input_width // 16
    output_flat_size = output_depth * output_height * output_width
    
    # Create output tensors
    conv_output = torch.empty((batch_size, output_flat_size, out_channels), 
                             dtype=in_3.dtype, device=in_3.device)
    position_output = torch.empty((1, output_flat_size, out_channels),
                                 dtype=in_3.dtype, device=in_3.device)
    
    # Block size for kernel
    BLOCK_SIZE = 1024
    num_programs = (output_flat_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel for conv3d + flatten + transpose
    fused_conv3d_flatten_transpose_kernel[(num_programs, 1, 1)](
        input_ptr=in_3,
        weight_ptr=in_1, 
        bias_ptr=in_0,
        output_ptr=conv_output,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        input_depth=input_depth,
        input_height=input_height, 
        input_width=input_width,
        weight_depth=weight_depth,
        weight_height=weight_height,
        weight_width=weight_width,
        output_depth=output_depth,
        output_height=output_height,
        output_width=output_width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Handle position embeddings separately (detach + type conversion)
    # Note: in_2 has shape [1, 1568, 768], we need to match with conv_output [1, 1568, 768]
    position_intermediate = in_2.detach()
    position_output = position_intermediate.type_as(conv_output)
    
    return conv_output, position_output

def replacement_func():
    """Return the fused kernel function"""
    return fused_conv3d_flatten_transpose