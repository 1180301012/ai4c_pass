import torch
import triton
import triton.language as tl
import math

def pattern(in_0, in_1, in_2):
    """Match conv2d followed by flatten from dimension 2"""
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.flatten(conv2d, 2)
    return tmp_3

def replacement_args(in_0, in_1, in_2):
    """Extract arguments for the fused conv2d+flatten operation"""
    return (in_0, in_1, in_2)

@triton.jit
def fused_conv2d_flatten_kernel(
    bias_ptr,
    weight_ptr,
    input_ptr,
    output_ptr,
    N: tl.constexpr,
    C_out: tl.constexpr,
    C_in: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for 1x1 conv2d followed by flatten from dim 2"""
    pid = tl.program_id(0)
    
    # Each program handles one element in the flattened output
    # Total elements: N * C_out * (H*W)
    total_elements = N * C_out * (H * W)
    element_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Mask to handle boundary conditions
    mask = element_idx < total_elements
    
    # Convert element index to output coordinates
    # element_idx = n * (C_out * H * W) + c_out * (H * W) + h * W + w
    # We want: n, c_out, (h, w) -> flattened h*w position
    flat_hw_size = H * W
    n = element_idx // (C_out * flat_hw_size)
    remainder = element_idx % (C_out * flat_hw_size)
    c_out = remainder // flat_hw_size
    flat_hw_pos = remainder % flat_hw_size
    h = flat_hw_pos // W
    w = flat_hw_pos % W
    
    # Load bias
    bias_val = tl.load(bias_ptr + c_out, mask=element_idx < total_elements)
    
    # Compute the convolution output for the spatial position (h, w)
    # Since kernel is 1x1, we just multiply input[:, :, h, w] by weight and add bias
    conv_val = bias_val
    
    # Accumulate: sum over input channels C_in
    for c_in in range(C_in):
        input_idx = n * (C_in * H * W) + c_in * (H * W) + h * W + w
        weight_idx = c_out * (C_in * 1 * 1) + c_in * (1 * 1) + 0 * 1 + 0
        
        input_val = tl.load(input_ptr + input_idx, mask=element_idx < total_elements)
        weight_val = tl.load(weight_ptr + weight_idx, mask=element_idx < total_elements)
        
        conv_val += input_val * weight_val
    
    # Store the result (already flattened)
    tl.store(output_ptr + element_idx, conv_val, mask=mask)

@torch.fx.wrap
def fused_conv2d_flatten_func(bias, weight, input_):
    """Fused function for conv2d + flatten operation"""
    # Get input dimensions
    N, C_in, H, W = input_.shape
    
    # Output channels from bias size
    C_out = bias.shape[0]
    
    # Calculate output shape: [N, C_out, H*W]
    output_shape = (N, C_out, H * W)
    output_size = N * C_out * H * W
    
    # Choose block size
    BLOCK_SIZE = 128
    
    # Calculate grid size
    grid_size = (output_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty(output_size, dtype=input_.dtype, device=input_.device)
    
    # Launch kernel - use tuple for grid size even in 1D
    fused_conv2d_flatten_kernel[(grid_size,)](
        bias_ptr=bias,
        weight_ptr=weight,
        input_ptr=input_,
        output_ptr=output,
        N=N,
        C_out=C_out,
        C_in=C_in,
        H=H,
        W=W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape to [N, C_out, H, W] -> [N, C_out, H*W]
    return output.view(N, C_out, H * W)

def replacement_func():
    """Return the fused function"""
    return fused_conv2d_flatten_func