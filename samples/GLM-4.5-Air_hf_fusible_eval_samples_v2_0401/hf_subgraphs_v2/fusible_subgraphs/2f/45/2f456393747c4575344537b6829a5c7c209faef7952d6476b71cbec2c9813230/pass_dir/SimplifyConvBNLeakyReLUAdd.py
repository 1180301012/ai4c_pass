import torch
import triton
import triton.language as tl

# Pattern matching function - must mirror the computation in model.py exactly
def pattern(in_6, in_4, in_0, in_1, in_3, in_2, in_5):
    """
    Pattern: Conv2D + BatchNorm + LeakyReLU + Residual Addition
    This matches the exact computation sequence from the model
    """
    conv2d = torch.conv2d(in_6, in_4, None, (1, 1), (1, 1), (1, 1), 1)
    tmp_6 = torch.nn.functional.batch_norm(conv2d, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_7 = torch.nn.functional.leaky_relu(tmp_6, 0.01, True)
    tmp_8 = tmp_7 + in_5
    return tmp_8

# Argument extraction function
def replacement_args(in_6, in_4, in_0, in_1, in_3, in_2, in_5):
    return (in_6, in_4, in_0, in_1, in_3, in_2, in_5)

# Triton kernel for BatchNorm + LeakyReLU + Addition (fallback implementation)
@triton.jit
def simplified_conv_bn_leakyrelu_add_kernel(
    conv_out_ptr,             # Conv output [N, C, H, W]
    running_mean_ptr,         # BN mean [C]
    running_var_ptr,          # BN var [C] 
    bn_weight_ptr,            # BN weight (gamma) [C]
    bn_bias_ptr,              # BN bias (beta) [C]
    residual_ptr,             # Residual tensor [N, C, H, W]
    output_ptr,               # Output tensor [N, C, H, W]
    n_channels,               # Channels (C)
    n_batches,                # Batch size (N)
    height,                   # Height (H)
    width,                    # Width (W)
    eps: tl.constexpr,        # BN epsilon (1e-05)
    neg_slope: tl.constexpr,  # Leaky ReLU negative slope (0.01)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Calculate coordinates
    batch = pid // (height * width)
    flat_idx = pid % (height * width)
    h = flat_idx // width
    w = flat_idx % width
    c = tl.program_id(1)
    
    # Only process if within bounds
    if batch >= n_batches or c >= n_channels or h >= height or w >= width:
        return
    
    # Load conv output, BN parameters, and residual
    conv_offset = batch * n_channels * height * width + c * height * width + h * width + w
    conv_val = tl.load(conv_out_ptr + conv_offset)
    
    bn_mean = tl.load(running_mean_ptr + c)
    bn_var = tl.load(running_var_ptr + c)
    bn_gamma = tl.load(bn_weight_ptr + c)
    bn_beta = tl.load(bn_bias_ptr + c)
    
    residual_offset = batch * n_channels * height * width + c * height * width + h * width + w
    residual_val = tl.load(residual_ptr + residual_offset)
    
    # Batch normalization: y = γ * (x - μ) / √(σ² + ε) + β
    var_plus_eps = bn_var + eps
    inv_std_dev = tl.math.rsqrt(var_plus_eps)
    normalized = (conv_val - bn_mean) * inv_std_dev
    bn_output = normalized * bn_gamma + bn_beta
    
    # Leaky ReLU: f(x) = x if x > 0, else neg_slope * x
    relu_output = tl.where(bn_output > 0, bn_output, bn_output * neg_slope)
    
    # Add residual
    final_output = relu_output + residual_val
    
    # Store result
    output_offset = batch * n_channels * height * width + c * height * width + h * width + w
    tl.store(output_ptr + output_offset, final_output)

# Simplified function that uses PyTorch conv2d but optimizes the rest
@torch.fx.wrap
def simplified_conv_bn_leakyrelu_add(x, weight, running_mean, running_var, bn_weight, bn_bias, residual):
    """
    Simplified fusion: Use PyTorch conv2d, then fuse BN + LeakyReLU + Addition with Triton
    """
    # Get tensor shapes
    N, C_in, H, W = x.shape
    O, I, KH, KW = weight.shape
    
    # Perform standard PyTorch convolution (let PyTorch handle the complex optimization)
    conv_out = torch.conv2d(x, weight, None, (1, 1), (1, 1), (1, 1), 1)
    
    # Verify shapes
    assert conv_out.shape == residual.shape, f"Conv output {conv_out.shape} doesn't match residual {residual.shape}"
    assert O == bn_weight.shape[0], "Output channels must match BN parameters"
    
    # Output tensor
    output = torch.empty_like(conv_out)
    
    # Launch Triton kernel for BN + LeakyReLU + Addition
    grid = (N * H * W, O)  # Grid dimensions: (batch_height_width, output_channels)
    
    simplified_conv_bn_leakyrelu_add_kernel[grid](
        conv_out_ptr=conv_out,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        bn_weight_ptr=bn_weight,
        bn_bias_ptr=bn_bias,
        residual_ptr=residual,
        output_ptr=output,
        n_channels=O,
        n_batches=N,
        height=H,
        width=W,
        eps=1e-05,
        neg_slope=0.01,
        BLOCK_SIZE_M=256,
        BLOCK_SIZE_N=128
    )
    
    return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return simplified_conv_bn_leakyrelu_add