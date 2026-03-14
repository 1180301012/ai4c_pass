import torch
import triton
import triton.language as tl

def pattern(x, weight, running_mean, running_var, weight_norm, bias_norm):
    # Conv2D + View + BatchNorm pattern - matching original computation structure
    conv_out = torch.conv2d(input=x, weight=weight, groups=512)
    viewed_out = conv_out.view(1, 512, 64, 64)  # This is the actual computation from original
    bn_out = torch.nn.functional.batch_norm(viewed_out, running_mean, running_var, 
                                          weight_norm, bias_norm, False, 0.1, 1e-05)
    return conv_out, viewed_out, bn_out

def replacement_args(x, weight, running_mean, running_var, weight_norm, bias_norm):
    return (x, weight, running_mean, running_var, weight_norm, bias_norm)

@triton.jit
def fused_conv_bn_kernel(
    x_ptr, weight_ptr, conv_out_ptr, bn_out_ptr,
    running_mean_ptr, running_var_ptr,
    weight_norm_ptr, bias_norm_ptr,
    N, C_in, H_in, W_in,
    C_out, H_out, W_out,
    kernel_size, stride, padding,
    groups, eps: tl.constexpr, momentum: tl.constexpr
):
    pid = tl.program_id(0)
    num_programs = tl.cdiv(N * C_out * H_out * W_out, 1024)
    
    if pid >= num_programs:
        return
    
    # Each program handles multiple output elements
    total_elements = N * C_out * H_out * W_out
    block_start = pid * 1024
    offsets = block_start + tl.arange(0, 1024)
    mask = offsets < total_elements
    
    # Calculate output indices
    n = offsets // (C_out * H_out * W_out)
    c = (offsets % (C_out * H_out * W_out)) // (H_out * W_out)
    h = (offsets % (H_out * W_out)) // W_out
    w = offsets % W_out
    
    # Calculate input indices for grouped convolution
    group_id = c // (C_out // groups)
    local_c_in = (c % (C_out // groups)) * (C_in // groups) + group_id
    local_c_out = c
    
    # Convolution indices
    x_h = h * stride[0] - padding[0]
    x_w = w * stride[1] - padding[1]
    
    # Load weight tensor (1x7x7 for each group)
    weight_offset = group_id * kernel_size[0] * kernel_size[1]
    weight_indices = (tl.arange(0, kernel_size[0] * kernel_size[1]) + weight_offset).to(tl.int32)
    weight_values = tl.load(weight_ptr + weight_indices, mask=(tl.arange(0, kernel_size[0] * kernel_size[1]) < (kernel_size[0] * kernel_size[1])).to(tl.int32))
    weight_values = tl.reshape(weight_values, [kernel_size[0], kernel_size[1]])
    
    # Convolution computation
    conv_value = 0.0
    kh = tl.arange(0, kernel_size[0])
    kw = tl.arange(0, kernel_size[1])
    
    for kh_idx in range(kernel_size[0]):
        for kw_idx in range(kernel_size[1]):
            h_idx = x_h + kh_idx
            w_idx = x_w + kw_idx
            
            if (0 <= h_idx < H_in) and (0 <= w_idx < W_in):
                x_offset = n * C_in * H_in * W_in + local_c_in * H_in * W_in + h_idx * W_in + w_idx
                x_val = tl.load(x_ptr + x_offset, mask=True)
                conv_value += x_val * weight_values[kh_idx, kw_idx]
    
    # Load batch norm parameters
    mean = tl.load(running_mean_ptr + c)
    var = tl.load(running_var_ptr + c)
    gamma = tl.load(weight_norm_ptr + c)
    beta = tl.load(bias_norm_ptr + c)
    
    # Batch norm computation
    inv_var = 1.0 / tl.sqrt(var + eps)
    bn_value = gamma * (conv_value - mean) * inv_var + beta
    
    # Store convolution result
    output_offset = n * C_out * H_out * W_out + local_c_out * H_out * W_out + h * W_out + w
    tl.store(conv_out_ptr + output_offset, conv_value)
    
    # Store batch norm result
    tl.store(bn_out_ptr + offsets, bn_value, mask=mask)

@torch.fx.wrap
def fused_conv_bn(x, weight, running_mean, running_var, weight_norm, bias_norm):
    N, C_in, H_in, W_in = x.shape
    C_out, K_h, K_w = weight.shape
    H_out = 64
    W_out = 64
    
    # Output tensors
    conv_out = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)
    viewed_out = conv_out  # The view operation will be optimized but we still need to return it
    bn_out = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)
    
    # Launch fused kernel
    fused_conv_bn_kernel[(triton.cdiv(N * C_out * H_out * W_out, 1024),)](
        x_ptr=x, weight_ptr=weight, conv_out_ptr=conv_out, bn_out_ptr=bn_out,
        running_mean_ptr=running_mean, running_var_ptr=running_var,
        weight_norm_ptr=weight_norm, bias_norm_ptr=bias_norm,
        N=N, C_in=C_in, H_in=H_in, W_in=W_in,
        C_out=C_out, H_out=H_out, W_out=W_out,
        kernel_size=(7, 7), stride=(1, 1), padding=(3, 3),
        groups=512, eps=1e-05, momentum=0.1
    )
    
    return conv_out, viewed_out, bn_out

def replacement_func():
    return fused_conv_bn