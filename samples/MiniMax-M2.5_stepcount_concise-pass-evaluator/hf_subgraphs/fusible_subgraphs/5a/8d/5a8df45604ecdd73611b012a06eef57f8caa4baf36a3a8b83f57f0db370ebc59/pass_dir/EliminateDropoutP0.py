import torch
import triton
import triton.language as tl


def pattern(conv_input, weight, bias):
    """
    Pattern that matches: conv2d followed by dropout(p=0.0).
    
    When dropout p=0.0, it's a no-op and returns input unchanged.
    This optimization eliminates the unnecessary dropout operation.
    The pattern matches: conv -> dropout, and replaces with just conv.
    """
    # Conv2d: 1x1 conv
    conv_out = torch.conv2d(conv_input, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    
    # Dropout with p=0.0 is a no-op - it just returns the input
    # We return conv_out since it equals dropout_out when p=0.0
    dropout_out = torch.nn.functional.dropout(conv_out, 0.0, False, False)
    
    # Return dropout_out which is what the rest of the computation uses
    return dropout_out


def replacement_args(conv_input, weight, bias):
    """
    Extract the arguments needed for the optimized replacement.
    We skip dropout entirely since p=0.0.
    """
    return (conv_input, weight, bias)


# Triton kernel for 1x1 conv (effectively a simple matrix multiply per spatial position)
@triton.jit
def conv_1x1_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    B, C_in, H, W, C_out,
    stride_in_b, stride_in_c, stride_in_h, stride_in_w,
    stride_w_o, stride_w_i,
    BLOCK_SIZE: tl.constexpr
):
    """Triton kernel for 1x1 conv - optimized for the specific case."""
    pid = tl.program_id(0)
    num_elements = B * C_out * H * W
    
    if pid >= num_elements:
        return
    
    # Calculate indices
    idx = pid
    c_out = idx % C_out
    idx = idx // C_out
    h = idx % H
    idx = idx // H
    w = idx % W
    b = idx // W
    
    # Compute output offset
    offset = b * stride_in_b + c_out * stride_in_w + h * stride_in_h + w * stride_in_w
    
    # Load input - for 1x1 conv, we need to sum over C_in channels
    # This is a matmul: output[b,c,h,w] = sum over c_in of input[b,c_in,h,w] * weight[c_out,c_in]
    result = 0.0
    for c_in in range(C_in):
        # Compute input offset for [b, c_in, h, w]
        in_offset = b * stride_in_b + c_in * stride_in_c + h * stride_in_h + w * stride_in_w
        input_val = tl.load(input_ptr + in_offset)
        
        # Compute weight offset for [c_out, c_in, 0, 0]
        w_offset = c_out * stride_w_o + c_in * stride_w_i
        weight_val = tl.load(weight_ptr + w_offset)
        
        result = result + input_val * weight_val
    
    # Add bias
    bias_val = tl.load(bias_ptr + c_out)
    result = result + bias_val
    
    # Store
    tl.store(output_ptr + offset, result)


@torch.fx.wrap
def triton_conv_1x1(conv_input, weight, bias):
    """Triton-accelerated 1x1 conv that skips the dropout."""
    B, C_in, H, W = conv_input.shape
    C_out = weight.shape[0]
    
    # Allocate output
    output = torch.empty((B, C_out, H, W), dtype=conv_input.dtype, device=conv_input.device)
    
    # Launch kernel
    num_elements = B * C_out * H * W
    BLOCK_SIZE = 1024
    num_programs = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    conv_1x1_kernel[(num_programs,)](
        conv_input, weight, bias, output,
        B, C_in, H, W, C_out,
        conv_input.stride(0), conv_input.stride(1), conv_input.stride(2), conv_input.stride(3),
        weight.stride(0), weight.stride(1),
        BLOCK_SIZE
    )
    
    return output


def replacement_func():
    """
    Return the optimized replacement function that skips dropout.
    """
    return triton_conv_1x1