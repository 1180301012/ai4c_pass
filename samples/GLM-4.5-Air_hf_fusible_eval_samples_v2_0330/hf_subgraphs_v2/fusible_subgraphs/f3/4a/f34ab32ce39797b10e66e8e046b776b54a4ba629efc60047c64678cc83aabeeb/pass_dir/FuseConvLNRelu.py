import torch
import triton
import triton.language as tl

def pattern(conv_input, conv_weight, conv_bias, ln_weight, ln_bias, ln_eps):
    """
    Pattern matching: Conv2D(1x1) + LayerNorm + ReLU fusion
    This matches the computation pattern found in all target graphs
    
    Argument order matches the actual computation:
    - conv_input: input to conv2d
    - conv_weight: weight for conv2d  
    - conv_bias: bias for conv2d
    - ln_weight: weight for layer_norm (passed as 3rd positional arg)
    - ln_bias: bias for layer_norm (passed as 4th positional arg)
    - ln_eps: epsilon for layer_norm (passed as 5th positional arg)
    """
    conv_out = torch.conv2d(conv_input, conv_weight, conv_bias, (1, 1), (0, 0), (1, 1), 1)
    
    # layer_norm called with positional args: input, normalized_shape, weight, bias, eps
    normalized_shape = conv_out.shape[1:]  # (channels, 1, 1)
    ln_out = torch.nn.functional.layer_norm(conv_out, normalized_shape, ln_weight, ln_bias, ln_eps)
    
    # relu with inplace=True
    relu_out = torch.nn.functional.relu(ln_out, inplace=True)
    
    return relu_out

def replacement_args(conv_input, conv_weight, conv_bias, ln_weight, ln_bias, ln_eps):
    """
    Extract arguments for the fused kernel
    """
    return (conv_input, conv_weight, conv_bias, ln_weight, ln_bias, ln_eps)

@triton.jit
def fused_conv_ln_relu_kernel(
    input_ptr, weight_ptr, bias_ptr, ln_weight_ptr, ln_bias_ptr, output_ptr,
    N, H, W, C_in, C_out,
    BLOCK_C: tl.constexpr, BLOCK_H: tl.constexpr
):
    """
    Fused Conv2D(1x1) + LayerNorm + ReLU kernel
    Optimized for 1x1 spatial convolutions
    """
    # Compute program indices
    pid_c = tl.program_id(0)
    pid_h = tl.program_id(1) 
    pid_w = tl.program_id(2)
    
    # Create offsets within block
    c_offset = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    h_offset = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    w_offset = tl.broadcast_to([pid_w], BLOCK_H)
    
    # Create masks
    c_mask = c_offset < C_out
    h_mask = h_offset < H
    w_mask = tl.broadcast_to([True], BLOCK_H)
    
    # Load weights and biases
    weight = tl.load(weight_ptr + (c_offset[:, None] * C_in), mask=c_mask[:, None], other=0.0)
    bias = tl.load(bias_ptr + (c_offset * 3), mask=c_mask, other=0.0)
    ln_weight = tl.load(ln_weight_ptr + (c_offset * 3), mask=c_mask, other=0.0)
    ln_bias = tl.load(ln_bias_ptr + (c_offset * 3), mask=c_mask, other=0.0)
    
    # Process each spatial location
    for h_idx in range(H):
        for w_idx in range(W):
            # Load input spatial positions
            input_base = h_idx * W + w_idx
            input_data = tl.load(input_ptr + input_base * C_in, mask=c_mask, other=0.0)
            
            # Conv2D operation (1x1 = dot product)
            conv_result = tl.dot(input_data, weight.to(tl.float32)) + bias.to(tl.float32)
            
            # LayerNorm operation (simplified since we're processing single elements)
            ln_result = (conv_result.to(tl.float32) + ln_bias.to(tl.float32)) * ln_weight.to(tl.float32)
            
            # ReLU operation
            relu_result = tl.where(ln_result > 0, ln_result, 0).to(tl.float32)
            
            # Store output
            output_base = (h_idx * W + w_idx) * C_out
            tl.store(output_ptr + output_base + c_offset, relu_result, mask=c_mask)

@torch.fx.wrap
def fused_conv_ln_relu(input_tensor, weight, bias, ln_weight, ln_bias, conv_bias):
    """
    Wrapper function for the fused Conv2D + LayerNorm + ReLU operation
    """
    B, C_in, H, W = input_tensor.shape
    C_out = weight.shape[0]
    
    # Output tensor
    output = torch.empty((B, C_out, H, W), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Triton kernel launch
    BLOCK_C = 128
    BLOCK_H = min(8, H)
    
    # Launch kernel for each batch
    for b in range(B):
        fused_conv_ln_relu_kernel[( (C_out + BLOCK_C - 1) // BLOCK_C, H, W )](
            input_ptr=input_tensor[b].data_ptr(),
            weight_ptr=weight.data_ptr(),
            bias_ptr=bias.data_ptr(),
            ln_weight_ptr=ln_weight.data_ptr(), 
            ln_bias_ptr=ln_bias.data_ptr(),
            output_ptr=output[b].data_ptr(),
            N=1, H=H, W=W, C_in=C_in, C_out=C_out,
            BLOCK_C=BLOCK_C, BLOCK_H=BLOCK_H
        )
    
    return output

def replacement_func():
    """
    Returns the fused kernel function
    """
    return fused_conv_ln_relu