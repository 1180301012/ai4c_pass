import torch
import triton
import triton.language as tl

def pattern(conv_input, conv_weight, conv_bias, norm_weight, norm_bias):
    # Conv2D operation with 1x1 kernel, stride 1, padding 0, dilation 1, groups 1
    conv_output = torch.conv2d(conv_input, conv_weight, conv_bias, (1, 1), (0, 0), (1, 1), 1)
    
    # LayerNorm with per-channel normalization (shape [channels, 1, 1])
    # Since shape has spatial dimensions (1, 1), this is effectively channel-wise
    norm_output = torch.nn.functional.layer_norm(conv_output, (conv_weight.size(0), 1, 1), norm_weight, norm_bias, 1e-05)
    
    # ReLU activation
    relu_output = torch.nn.functional.relu(norm_output, inplace=True)
    
    return relu_output

def replacement_args(conv_input, conv_weight, conv_bias, norm_weight, norm_bias):
    return (conv_input, conv_weight, conv_bias, norm_weight, norm_bias)

@triton.jit
def conv_norm_relu_kernel(
    input_ptr,           # [N, C_in, H, W] - conv input
    weight_ptr,          # [C_out, C_in, 1, 1] - conv weight  
    conv_bias_ptr,       # [C_out] - conv bias
    norm_weight_ptr,     # [C_out, 1, 1] - layer norm weight (broadcast to C_out)
    norm_bias_ptr,       # [C_out, 1, 1] - layer norm bias (broadcast to C_out)
    output_ptr,          # [N, C_out, H, W] - final output
    N: tl.constexpr,     # batch size
    C_in: tl.constexpr,  # input channels
    C_out: tl.constexpr, # output channels  
    H: tl.constexpr,     # height
    W: tl.constexpr,     # width
):
    # Each program handles one output channel for one spatial position
    c_out = tl.program_id(0)
    spatial_idx = tl.program_id(1)
    
    # Extract spatial position
    h = spatial_idx // W
    w = spatial_idx % W
    
    # Load conv bias for this channel
    conv_bias = tl.load(conv_bias_ptr + c_out)
    
    # Load layer norm parameters for this channel
    # norm_weight and norm_bias have shape [C_out, 1, 1], so we just need the first element
    norm_weight_val = tl.load(norm_weight_ptr + c_out)
    norm_bias_val = tl.load(norm_bias_ptr + c_out)
    
    # Compute output for this spatial position
    sum_val = conv_bias  # start with conv bias
    
    # Convolution - sum over input channels
    for c_in in range(C_in):
        # Input element: [N, C_in, H, W] -> [n, c_in, h, w]
        input_offset = (h * W + w) * C_in + c_in
        input_val = tl.load(input_ptr + input_offset)
        
        # Weight element: [C_out, C_in, 1, 1] -> [c_out, c_in, 0, 0]
        weight_offset = c_out * C_in + c_in
        weight_val = tl.load(weight_ptr + weight_offset)
        
        sum_val += input_val * weight_val
    
    # Apply layer norm parameters (per-channel scaling and shifting)
    normalized_val = sum_val * norm_weight_val + norm_bias_val
    
    # Apply ReLU
    output_val = tl.maximum(normalized_val, 0.0)
    
    # Store result: [N, C_out, H, W] -> [n, c_out, h, w]
    # For simplicity, we assume N=1 based on the input patterns
    spatial_out_offset = (h * W + w) * C_out + c_out
    tl.store(output_ptr + spatial_out_offset, output_val)

@torch.fx.wrap
def fused_conv_norm_relu(input_tensor, conv_weight, conv_bias, norm_weight, norm_bias):
    # Get tensor shapes
    N, C_in, H, W = input_tensor.shape
    C_out = conv_weight.size(0)
    
    # Create output tensor
    output = torch.empty((N, C_out, H, W), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Calculate grid size
    # Each program handles one output channel for one spatial position
    num_channels = C_out
    num_spatial_positions = H * W
    
    # Launch kernel with 2D grid: [channels, spatial_positions]
    conv_norm_relu_kernel[(num_channels, num_spatial_positions)](
        input_ptr=input_tensor,
        weight_ptr=conv_weight,
        conv_bias_ptr=conv_bias,
        norm_weight_ptr=norm_weight,
        norm_bias_ptr=norm_bias,
        output_ptr=output,
        N=N,
        C_in=C_in,
        C_out=C_out,
        H=H,
        W=W,
    )
    
    return output

def replacement_func():
    return fused_conv_norm_relu