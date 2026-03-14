import torch
import triton
import triton.language as tl

def pattern(x_input, skip_input, weight, bias, running_mean, running_var, bn_weight, bn_bias):
    """
    Complete fused pattern: conv2d + add + add + batch_norm + mean
    """
    # Convolution operation (1x1)
    conv_out = torch.conv2d(x_input, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    # Two additions forming residual connection: x_input + skip_input + conv_out
    residual_out = x_input + skip_input + conv_out
    # Batch normalization
    bn_out = torch.nn.functional.batch_norm(residual_out, running_mean, running_var, bn_weight, bn_bias, False, 0.1, 1e-05)
    # Mean reduction over spatial dimensions
    mean_out = bn_out.mean((2, 3), keepdim=True)
    return bn_out, mean_out

def replacement_args(x_input, skip_input, weight, bias, running_mean, running_var, bn_weight, bn_bias):
    return (x_input, skip_input, weight, bias, running_mean, running_var, bn_weight, bn_bias)

@triton.jit
def complete_fusion_kernel(
    x_input_ptr, skip_input_ptr, weight_ptr, bias_ptr,
    running_mean_ptr, running_var_ptr, bn_weight_ptr, bn_bias_ptr,
    bn_out_ptr, mean_out_ptr,    
    N, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = N * C * H * W
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Reshape offsets to 4D: [N, C, H, W]
    offset_w = offsets % W
    offset_h = (offsets // W) % H
    offset_c = (offsets // (W * H)) % C
    offset_n = offsets // (W * H * C)
    
    # Load input tensors
    x_val = tl.load(x_input_ptr + offsets, mask=mask, other=0.0)
    skip_val = tl.load(skip_input_ptr + offsets, mask=mask, other=0.0)
    
    # Load convolution parameters (1x1 conv, so channel-wise)
    channel_idx = offset_c % C
    conv_weight = tl.load(weight_ptr + channel_idx, mask=channel_idx < C)
    conv_bias = tl.load(bias_ptr + channel_idx, mask=channel_idx < C)
    
    # Apply convolution: Y = X * W + b (element-wise for 1x1 conv)
    conv_result = x_val * conv_weight + conv_bias
    
    # Fused addition: residual = x_input + skip_input + conv_result
    residual = x_val + skip_val + conv_result
    
    # Load batch normalization parameters
    bn_mean = tl.load(running_mean_ptr + channel_idx, mask=channel_idx < C)
    bn_var = tl.load(running_var_ptr + channel_idx, mask=channel_idx < C)
    bn_weight_val = tl.load(bn_weight_ptr + channel_idx, mask=channel_idx < C)
    bn_bias_val = tl.load(bn_bias_ptr + channel_idx, mask=channel_idx < C)
    
    # Batch normalization: y = (x - mean) / sqrt(var + eps) * weight + bias
    eps = 1e-05
    normalized = (residual - bn_mean) / tl.sqrt(bn_var + eps)
    bn_result = normalized * bn_weight_val + bn_bias_val
    
    # Store batch normalization output
    tl.store(bn_out_ptr + offsets, bn_result, mask=mask)
    
    # For mean reduction, we need to compute spatial mean per channel per batch
    # Use atomic operations to accumulate partial sums
    mean_idx = offset_n * C + offset_c
    if offset_w == 0 and offset_h == 0:  # Only one thread per spatial location contributes
        spatial_sum = 0.0
        for hw in range(H * W):
            hw_offset = offset_n * C * H * W + offset_c * H * W + hw
            if hw_offset < total_elements:
                spatial_sum += tl.load(bn_out_ptr + hw_offset, other=0.0)
        
        mean_val = spatial_sum / (H * W)
        # Store mean in output buffer [N, C]
        tl.store(mean_out_ptr + mean_idx, mean_val)

@torch.fx.wrap  
def complete_fusion_block(x_input, skip_input, weight, bias, running_mean, running_var, bn_weight, bn_bias):
    N, C, H, W = x_input.shape
    
    # Create output tensors
    bn_out = torch.empty_like(x_input)
    mean_out = torch.empty((N, C, 1, 1), dtype=x_input.dtype, device=x_input.device)
    
    BLOCK_SIZE = 1024
    total_elements = N * C * H * W
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Flatten mean output for easier atomic operations
    mean_flat = mean_out.squeeze()  # [N, C]
    
    # Launch kernel
    complete_fusion_kernel[(num_programs,)](
        x_input_ptr=x_input,
        skip_input_ptr=skip_input,
        weight_ptr=weight.squeeze(),  # Remove spatial dims for 1x1 conv
        bias_ptr=bias,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        bn_weight_ptr=bn_weight,
        bn_bias_ptr=bn_bias,
        bn_out_ptr=bn_out,
        mean_out_ptr=mean_flat,  # Pass flattened mean buffer
        N=N, C=C, H=H, W=W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape mean output back to [N, C, 1, 1]
    mean_out = mean_flat.reshape(N, C, 1, 1)
    
    return bn_out, mean_out

def replacement_func():
    return complete_fusion_block