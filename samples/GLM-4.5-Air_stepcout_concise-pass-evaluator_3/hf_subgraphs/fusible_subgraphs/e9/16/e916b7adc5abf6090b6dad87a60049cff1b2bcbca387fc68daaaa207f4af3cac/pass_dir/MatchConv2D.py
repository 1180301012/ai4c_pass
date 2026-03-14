import torch
import triton
import triton.language as tl

def pattern(x, weight, bias):
    # Simple conv2d pattern test
    return torch.conv2d(x, weight, bias, (1, 1), (0, 0), (1, 1), 1)

def replacement_args(x, weight, bias):
    return (x, weight, bias)

@triton.jit
def efficient_conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    N, C_in, C_out, H, W, BLOCK_SIZE: tl.constexpr,
):
    # Efficient block-level parallel conv2d
    pid = tl.program_id(0)
    
    # Number of elements that each program handles (BLOCK_SIZE)
    total_elements = N * C_out * H * W
    if pid * BLOCK_SIZE >= total_elements:
        return
    
    # Vectorized access pattern - each program handles BLOCK_SIZE consecutive elements
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Convert linear offsets to tensor coordinates
    n_coords = offsets // (C_out * H * W)
    remainder = offsets % (C_out * H * W)
    c_out_coords = remainder // (H * W)
    h_coords = (remainder // W) % H
    w_coords = remainder % W
    
    # Process each element in the block
    for i in range(BLOCK_SIZE):
        if offsets[i] >= total_elements:
            continue
            
        n = n_coords[i]
        c_out = c_out_coords[i]
        h = h_coords[i]
        w = w_coords[i]
        
        # Compute 1x1 convolution for this pixel
        conv_val = 0.0
        for c_in in range(C_in):
            # Load weight and input
            weight_offset = (c_out, c_in, 0, 0)
            input_offset = (n, c_in, h, w)
            
            weight_val = tl.load(weight_ptr + weight_offset, mask=mask[i], other=0.0)
            input_val = tl.load(input_ptr + input_offset, mask=mask[i], other=0.0)
            
            conv_val += weight_val * input_val
        
        # Add bias
        bias_offset = (c_out,)
        bias_val = tl.load(bias_ptr + bias_offset, mask=mask[i], other=0.0)
        conv_val += bias_val
        
        # Store result
        output_offset = (n, c_out, h, w)
        tl.store(output_ptr + output_offset, conv_val, mask=mask[i])

@torch.fx.wrap
def simple_conv2d(x, weight, bias):
    N, C_in, H, W = x.shape
    C_out = weight.shape[0]
    
    # Use efficient block processing - each program handles BLOCK_SIZE elements
    BLOCK_SIZE = 256  # Number of elements per CUDA program (tune for GPU performance)
    
    total_elements = N * C_out * H * W
    total_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty((N, C_out, H, W), dtype=x.dtype, device=x.device)
    
    # Launch optimized kernel
    efficient_conv2d_kernel[total_programs](
        x, weight, bias, output,
        N, C_in, C_out, H, W,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return simple_conv2d