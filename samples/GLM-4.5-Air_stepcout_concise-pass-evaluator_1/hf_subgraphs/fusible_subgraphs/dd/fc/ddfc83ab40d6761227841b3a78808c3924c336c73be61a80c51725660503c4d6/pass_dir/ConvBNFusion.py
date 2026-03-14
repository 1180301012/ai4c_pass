import torch
import triton
import triton.language as tl

def pattern(x, weight, bias, running_mean, running_var, bn_weight, bn_bias):
    """
    Conv2d + BatchNorm fusion pattern
    This pattern matches the sequence: conv -> batch normalization
    """
    # Convolution
    conv_out = torch.conv2d(x, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    # Batch normalization 
    bn_out = torch.nn.functional.batch_norm(conv_out, running_mean, running_var, bn_weight, bn_bias, False, 0.1, 1e-05)
    return bn_out

def replacement_args(x, weight, bias, running_mean, running_var, bn_weight, bn_bias):
    return (x, weight, bias, running_mean, running_var, bn_weight, bn_bias)

@triton.jit
def conv_bn_fused_kernel(
    x_ptr, weight_ptr, bias_ptr, 
    running_mean_ptr, running_var_ptr, bn_weight_ptr, bn_bias_ptr,
    out_ptr,
    N, C_out, H, W, C_in,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = N * C_out * H * W
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Reshape offsets to 4D: [N, C_out, H, W]
    offset_w = offsets % W
    offset_h = (offsets // W) % H
    offset_c_out = (offsets // (W * H)) % C_out
    offset_n = offsets // (W * H * C_out)
    
    # Load input tensor [N, C_in, H, W]
    x_val = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load convolution parameters [C_out]
    conv_weight = tl.load(weight_ptr + offset_c_out, mask=offset_c_out < C_out)
    conv_bias = tl.load(bias_ptr + offset_c_out, mask=offset_c_out < C_out)
    
    # Load batch norm parameters [C_out]
    bn_mean = tl.load(running_mean_ptr + offset_c_out, mask=offset_c_out < C_out)
    bn_var = tl.load(running_var_ptr + offset_c_out, mask=offset_c_out < C_out)
    bn_weight_val = tl.load(bn_weight_ptr + offset_c_out, mask=offset_c_out < C_out)
    bn_bias_val = tl.load(bn_bias_ptr + offset_c_out, mask=offset_c_out < C_out)
    
    # Fused computation: 
    # 1. Apply convolution (1x1 with groups=1)  
    # 2. Apply batch normalization
    # Formula: (conv(x * w + b) - mean) / sqrt(var + eps) * bn_weight + bn_bias
    
    conv_result = x_val * conv_weight + conv_bias
    eps = 1e-05
    
    # Batch normalization fused with convolution
    # We can optimize by pre-computing scale factors
    # scale = bn_weight / sqrt(var + eps)
    # bias = (conv_bias - mean * scale) + bn_bias
    
    # For simplicity, compute sequentially but in one kernel
    normalized = (conv_result - bn_mean) / tl.sqrt(bn_var + eps)
    final_result = normalized * bn_weight_val + bn_bias_val
    
    # Store result
    tl.store(out_ptr + offsets, final_result, mask=mask)

@torch.fx.wrap
def fused_conv_bn(x, weight, bias, running_mean, running_var, bn_weight, bn_bias):
    """Fused conv2d + batch normalization implementation"""
    N, C_in, H, W = x.shape
    C_out = weight.shape[0]  # Output channels
    
    output = torch.empty((N, C_out, H, W), dtype=x.dtype, device=x.device)
    
    # Choose block size based on tensor size for optimal performance
    total_elements = N * C_out * H * W
    if total_elements < 1000000:
        block_size = 1024
    elif total_elements < 10000000:
        block_size = 2048
    else:
        block_size = 4096
    
    num_programs = (total_elements + block_size - 1) // block_size
    
    conv_bn_fused_kernel[(num_programs,)](
        x_ptr=x,
        weight_ptr=weight.squeeze(),  # Remove spatial dims for 1x1 conv
        bias_ptr=bias,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        bn_weight_ptr=bn_weight,
        bn_bias_ptr=bn_bias,
        out_ptr=output,
        N=N, C_out=C_out, H=H, W=W, C_in=C_in,
        BLOCK_SIZE=block_size,
    )
    
    return output

def replacement_func():
    return fused_conv_bn