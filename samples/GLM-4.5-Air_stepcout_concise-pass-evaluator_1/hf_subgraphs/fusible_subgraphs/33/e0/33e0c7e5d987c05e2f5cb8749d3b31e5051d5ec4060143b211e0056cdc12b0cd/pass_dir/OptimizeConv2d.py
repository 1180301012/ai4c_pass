import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    """Pattern matches the model structure"""
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.hardtanh(in_3, 0.0, 6.0, False)
    return (tmp_3, tmp_2)

def replacement_args(in_0, in_1, in_2, in_3):
    """Extract arguments for the optimized kernel"""
    return (in_0, in_1, in_2, in_3)

@triton.jit
def optimized_conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    N, C_in, H, W, C_out,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    """Optimized Conv2D kernel for 1x1 convolution"""
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate output ranges
    m_start = pid * BLOCK_SIZE_M
    m_end = min(m_start + BLOCK_SIZE_M, C_out)
    
    if m_start >= C_out:
        return
    
    # Process one output channel at a time for simplicity
    for m in range(m_start, m_end):
        # Initialize output with bias
        bias_val = tl.load(bias_ptr + m)
        output_val = bias_val
        
        # Sum over input channels and spatial positions
        for c_in in range(C_in):
            weight_val = tl.load(weight_ptr + m * C_in + c_in)
            # For 1x1 convolution, sum over all positions in this channel
            base_input = input_ptr + c_in * H * W
            for h in range(H):
                for w in range(W):
                    input_val = tl.load(base_input + h * W + w)
                    output_val += weight_val * input_val
        
        # Store result
        base_output = output_ptr + m * H * W
        for h in range(H):
            for w in range(W):
                tl.store(base_output + h * W + w, output_val)

@triton.jit
def hardtanh_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    """Apply HardTanh: max(0, min(6, x)) using Triton"""
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    end_idx = min(start_idx + BLOCK_SIZE, n_elements)
    
    if start_idx >= n_elements:
        return
    
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply HardTanh: clip between 0 and 6
    y = tl.maximum(tl.minimum(x, 6.0), 0.0)
    
    # Store result
    tl.store(output_ptr + offsets, y, mask=mask)

@torch.fx.wrap
def optimized_conv2d(in_0, in_1, in_2, in_3):
    """Optimized Conv2D wrapper"""
    bias = in_0
    weight = in_1
    conv_input = in_2
    tanh_input = in_3
    
    # Get dimensions
    N, C_in, H, W = conv_input.shape
    C_out = weight.shape[0]
    
    # Create output tensors
    conv_output = torch.empty((N, C_out, H, W), dtype=conv_input.dtype, device=conv_input.device)
    tanh_output = torch.empty_like(tanh_input)
    
    # Block sizes for 1x1 convolution
    BLOCK_SIZE_M = 16  # Output channels per block
    
    # Calculate grid size for Conv2D
    grid_size = (C_out + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    
    # Launch Conv2D kernel
    optimized_conv2d_kernel[grid_size](
        conv_input, weight, bias, conv_output,
        N, C_in, H, W, C_out,
        BLOCK_SIZE_M, 1, 1
    )
    
    # Apply HardTanh using Triton kernel
    BLOCK_SIZE = 1024
    grid_size = (tanh_input.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE
    hardtanh_kernel[grid_size](
        tanh_input, tanh_output,
        tanh_input.numel(), BLOCK_SIZE
    )
    
    return (tanh_output, conv_output)

def replacement_func():
    """Returns the optimized function"""
    return optimized_conv2d