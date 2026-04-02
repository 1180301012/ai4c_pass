import torch
import triton
import triton.language as tl

def pattern(x, weight, running_mean, running_var, weight_bn, bias_bn, y):
    """
    Pattern matching for Conv2D + BatchNorm + Add operations
    x: input tensor to conv2d
    weight: conv2d weight
    running_mean, running_var: batch norm stats
    weight_bn, bias_bn: batch norm parameters
    y: tensor to add after batch norm
    """
    # Conv2D operation with specific parameters from the model
    conv_out = torch.conv2d(x, weight, None, (1, 1), (0, 0), (1, 1), 1)
    
    # BatchNorm operation  
    bn_out = torch.nn.functional.batch_norm(conv_out, running_mean, running_var, weight_bn, bias_bn, False, 0.1, 1e-05)
    
    # Addition operation
    out = bn_out + y
    return out

def replacement_args(x, weight, running_mean, running_var, weight_bn, bias_bn, y):
    return (x, weight, running_mean, running_var, weight_bn, bias_bn, y)

@triton.jit
def fused_conv_bn_kernel(
    x_ptr, weight_ptr, running_mean_ptr, running_var_ptr, 
    weight_bn_ptr, bias_bn_ptr, y_ptr, out_ptr,
    N, C, H, W, K, KH, KW,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program IDs
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    
    # Each program handles a portion of the output
    output_per_program = (N * C * H * W + num_pid - 1) // num_pid
    start_idx = pid * output_per_program
    end_idx = min((pid + 1) * output_per_program, N * C * H * W)
    
    # Precompute batch norm scale and bias
    scale = tl.load(weight_bn_ptr + tl.arange(0, C), mask=tl.arange(0, C) < C)
    bias = tl.load(bias_bn_ptr + tl.arange(0, C), mask=tl.arange(0, C) < C)
    mean = tl.load(running_mean_ptr + tl.arange(0, C), mask=tl.arange(0, C) < C)
    var = tl.load(running_var_ptr + tl.arange(0, C), mask=tl.arange(0, C) < C)
    
    # Precompute 1/sqrt(var + eps)
    eps = 1e-05
    var_inv_sqrt = tl.math.rsqrt(var + eps)
    scaled_scale = scale * var_inv_sqrt
    shifted_bias = bias - mean * var_inv_sqrt * scale
    
    # Process the assigned portion
    idx = start_idx
    while idx < end_idx:
        # Calculate indices
        n = idx // (C * H * W)
        c = (idx // (H * W)) % C
        h = (idx // W) % H
        w = idx % W
        
        # Convolution computation for this spatial location
        conv_val = 0.0
        for kh in range(KH):
            for kw in range(KW):
                for ci in range(C):
                    # Calculate input coordinates
                    ih = h * 1 + kh  # stride=1
                    iw = w * 1 + kw  # stride=1
                    
                    if ih < H and iw < W:
                        # Load input weight
                        weight_offset = (kh * KW + kw) * C + ci
                        conv_val += tl.load(x_ptr + n * C * H * W + ci * H * W + ih * W + iw) * tl.load(weight_ptr + weight_offset)
        
        # Apply batch normalization (scaled and shifted conv result + y value)
        y_val = tl.load(y_ptr + n * C * H * W + c * H * W + h * W + w)
        normalized_val = conv_val * scaled_scale[c] + shifted_bias[c]
        out_val = normalized_val + y_val
        
        # Store result
        tl.store(out_ptr + idx, out_val)
        idx += 1

@torch.fx.wrap
def fused_conv_bn(x, weight, running_mean, running_var, weight_bn, bias_bn, y):
    # Get tensor shapes
    N, C_in, H, W = x.shape
    K, C_out, KH, KW = weight.shape
    
    # Create output tensor
    out = torch.empty(N, C_out, H, W, dtype=x.dtype, device=x.device)
    
    # Launch kernel
    block_size = 1024
    num_programs = (N * C_out * H * W + block_size - 1) // block_size
    
    fused_conv_bn_kernel[(num_programs,)](
        x_ptr=x,
        weight_ptr=weight,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_bn_ptr=weight_bn,
        bias_bn_ptr=bias_bn,
        y_ptr=y,
        out_ptr=out,
        N=N, C=C_out, H=H, W=W,
        K=KH, KH=KH, KW=KW,
        BLOCK_SIZE=block_size,
    )
    
    return out

def replacement_func():
    return fused_conv_bn