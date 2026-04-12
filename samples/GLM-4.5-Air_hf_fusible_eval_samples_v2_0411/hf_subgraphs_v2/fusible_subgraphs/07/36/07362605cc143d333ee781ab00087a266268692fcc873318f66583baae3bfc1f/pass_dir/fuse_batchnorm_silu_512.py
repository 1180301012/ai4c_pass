import torch
import triton
import triton.language as tl

def pattern(input_tensor, running_mean, running_var, weight, bias):
    """Match: reshape -> batch_norm -> silu pattern for 512 channels"""
    # Match the exact operations from the computation graph for 512 channels
    tmp_4 = input_tensor.reshape(1, 512, 8, 8)  # 4, 128, 64 -> 1, 512, 8, 8
    
    # Match exact batch_norm signature: input, running_mean, running_var, weight, bias, training, momentum, eps
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    
    # Match exact silu signature with inplace=True
    tmp_6 = torch.nn.functional.silu(tmp_5, inplace=True)
    return tmp_6

def replacement_args(input_tensor, running_mean, running_var, weight, bias):
    """Extract arguments for the fused kernel"""
    return (input_tensor, running_mean, running_var, weight, bias, 512, 8, 8)

@triton.jit
def fused_batchnorm_silu_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    eps: tl.constexpr,
    momentum: tl.constexpr,
    feature_dim: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused BatchNorm + SiLU kernel"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load parameters (broadcast to match input)
    if weight_ptr is not None:
        weight = tl.load(weight_ptr, mask=mask[:feature_dim], other=1.0)
        weight = tl.broadcast_to(weight, [height, width, feature_dim])
    else:
        weight = 1.0
        
    if bias_ptr is not None:
        bias = tl.load(bias_ptr, mask=mask[:feature_dim], other=0.0)
        bias = tl.broadcast_to(bias, [height, width, feature_dim])
    else:
        bias = 0.0
        
    if running_mean_ptr is not None:
        running_mean = tl.load(running_mean_ptr, mask=mask[:feature_dim], other=0.0)
        running_mean = tl.broadcast_to(running_mean, [height, width, feature_dim])
    else:
        running_mean = 0.0
        
    if running_var_ptr is not None:
        running_var = tl.load(running_var_ptr, mask=mask[:feature_dim], other=1.0)
        running_var = tl.broadcast_to(running_var, [height, width, feature_dim])
    else:
        running_var = 1.0
    
    # Fuse BatchNorm and SiLU operations
    # y = (x - mean) / sqrt(var + eps) * weight + bias
    # fused_out = silu(y) = y * sigmoid(y)
    
    # BatchNorm computation
    normalized = (x - running_mean) / tl.sqrt(running_var + eps)
    bn_out = normalized * weight + bias
    
    # SiLU (Swish) activation: silu(x) = x * sigmoid(x)
    silu_out = bn_out * (1.0 / (1.0 + tl.exp(-bn_out)))
    
    # Store result
    tl.store(out_ptr + offsets, silu_out, mask=mask)

@torch.fx.wrap
def fused_batchnorm_silu(input_tensor, running_mean, running_var, weight, bias, feature_dim, height, width):
    """Wrapper function for the fused kernel"""
    N, C, H, W = 1, feature_dim, height, width
    n_elements = N * C * H * W
    
    # Choose optimal block size
    if n_elements < 1024:
        BLOCK_SIZE = 64
    elif n_elements < 16384:
        BLOCK_SIZE = 256
    else:
        BLOCK_SIZE = 1024
    
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty((1, feature_dim, height, width), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel with proper parameter handling
    kernel_args = {
        'x_ptr': input_tensor,
        'out_ptr': out,
        'n_elements': n_elements,
        'eps': 1e-05,
        'momentum': 0.1,
        'feature_dim': feature_dim,
        'height': height,
        'width': width,
        'BLOCK_SIZE': BLOCK_SIZE,
    }
    
    # Handle None parameters (for cases where they might be None)
    if weight is None:
        kernel_args['weight_ptr'] = None
    else:
        kernel_args['weight_ptr'] = weight
        
    if bias is None:
        kernel_args['bias_ptr'] = None
    else:
        kernel_args['bias_ptr'] = bias
        
    if running_mean is None:
        kernel_args['running_mean_ptr'] = None
    else:
        kernel_args['running_mean_ptr'] = running_mean
        
    if running_var is None:
        kernel_args['running_var_ptr'] = None
    else:
        kernel_args['running_var_ptr'] = running_var
    
    fused_batchnorm_silu_kernel[(num_programs,)](**kernel_args)
    
    return out

def replacement_func():
    """Return the fused kernel function"""
    return fused_batchnorm_silu