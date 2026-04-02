import torch
import triton
import triton.language as tl
import math

def pattern(conv_input, weights, bias, add_input):
    """
    Pattern matching: Conv2D + Dropout(p=0.0) + Add
    The dropout with p=0.0 is essentially a no-op and should be eliminated.
    """
    # Perform 2D convolution with specific parameters from the model
    conv_result = torch.conv2d(conv_input, weights, bias, (1, 1), (0, 0), (1, 1), 1)
    
    # Dropout with p=0.0 - this is essentially a no-op operation
    dropout_result = torch.nn.functional.dropout(conv_result, 0.0, False, False)
    
    # Element-wise addition
    final_result = dropout_result + add_input
    
    return (final_result,)

def replacement_args(conv_input, weights, bias, add_input):
    """Extract arguments needed for the replacement kernel"""
    return (conv_input, weights, bias, add_input)

@triton.jit
def fused_conv2d_add_kernel(
    input_ptr,  # [N, C_in, H, W] = [1, 256, 4, 256]
    weight_ptr,  # [C_out, C_in, KH, KW] = [128, 256, 1, 1]
    bias_ptr,  # [C_out] = [128]
    add_input_ptr,  # [N, C_out, H, W] = [1, 128, 4, 256]
    output_ptr,  # [N, C_out, H, W] = [1, 128, 4, 256]
    N: tl.constexpr,
    C_in: tl.constexpr,
    H_in: tl.constexpr,
    W_in: tl.constexpr,
    C_out: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused Conv2D + Add operation, eliminating no-op dropout"""
    
    pid = tl.program_id(0)
    total_elements = N * C_out * H_out * W_out
    
    # Calculate each thread's workload
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    elements_per_program = (total_elements + num_programs - 1) // num_programs
    
    start_idx = pid * elements_per_program
    end_idx = min((pid + 1) * elements_per_program, total_elements)
    
    for idx in range(start_idx, end_idx):
        # Convert linear index to 4D coordinates: [N, C_out, H, W]
        w = idx % W_out
        h = (idx // W_out) % H_out
        c = (idx // (W_out * H_out)) % C_out
        n = idx // (W_out * H_out * C_out)
        
        # For 1x1 convolution with stride (1,1) and padding (0,0),
        # the output coordinates map directly to input coordinates for height/width
        hw_in_start = (h * 1)  # stride_h=1
        hw_in_end = min(hw_in_start + KH, H_in)
        ww_in_start = (w * 1)  # stride_w=1  
        ww_in_end = min(ww_in_start + KW, W_in)
        
        # Accumulate convolution result
        conv_val = 0.0
        if hw_in_start < hw_in_end and ww_in_start < ww_in_end:
            # For 1x1 convolution, we just multiply corresponding spatial positions
            for ci in range(C_in):
                # Load input value ([n, ci, h, w] -> [n, 256, 4, 256])
                input_offset = n * C_in * H_in * W_in + ci * H_in * W_in + h * W_in + w
                input_val = tl.load(input_ptr + input_offset)
                
                # Load weight value ([c, ci, 0, 0] -> [128, 256, 1, 1])
                weight_offset = c * C_in * KH * KW + ci * KH * KW + 0 * KW + 0
                weight_val = tl.load(weight_ptr + weight_offset)
                
                conv_val += input_val * weight_val
        
        # Add bias
        bias_offset = c
        bias_val = tl.load(bias_ptr + bias_offset)
        conv_val += bias_val
        
        # Load add input ([n, c, h, w] -> [1, 128, 4, 256])
        add_input_offset = n * C_out * H_out * W_out + c * H_out * W_out + h * W_out + w
        add_val = tl.load(add_input_ptr + add_input_offset)
        
        # Final result: conv + add_input
        final_val = conv_val + add_val
        
        # Store result
        output_offset = n * C_out * H_out * W_out + c * H_out * W_out + h * W_out + w
        tl.store(output_ptr + output_offset, final_val)

@torch.fx.wrap
def fused_conv2d_add_triton(conv_input, weights, bias, add_input):
    """
    Triton kernel implementing fused Conv2D + Add with eliminated no-op dropout
    """
    # Get tensor shapes from input arguments
    N, C_in, H_in, W_in = conv_input.shape
    C_out, _, KH, KW = weights.shape
    H_out, W_out = H_in, W_in  # Since stride=(1,1) and padding=(0,0)
    
    # Output shape should be same as add_input for element-wise addition
    expected_shape = (N, C_out, H_out, W_out)
    assert add_input.shape == expected_shape, f"Add input shape mismatch: expected {expected_shape}, got {add_input.shape}"
    
    # Create output tensor
    output = torch.empty(expected_shape, dtype=conv_input.dtype, device=conv_input.device)
    
    # Calculate optimal block size based on tensor sizes
    total_elements = N * C_out * H_out * W_out
    BLOCK_SIZE = 1024  # Optimal for this tensor size
    
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch Triton kernel
    fused_conv2d_add_kernel[(num_programs,)](
        input_ptr=conv_input,
        weight_ptr=weights,
        bias_ptr=bias,
        add_input_ptr=add_input,
        output_ptr=output,
        N=N, C_in=C_in, H_in=H_in, W_in=W_in,
        C_out=C_out, KH=KH, KW=KW,
        H_out=H_out, W_out=W_out,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return (output,)

def replacement_func():
    """Replacement function - returns the kernel wrapper"""
    return fused_conv2d_add_triton