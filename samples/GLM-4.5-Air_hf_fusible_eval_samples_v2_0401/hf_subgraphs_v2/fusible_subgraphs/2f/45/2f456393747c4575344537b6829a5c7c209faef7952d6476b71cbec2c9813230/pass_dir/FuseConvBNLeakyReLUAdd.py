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

# Triton kernel for fused Conv2D + BatchNorm + LeakyReLU + Addition
@triton.jit
def fused_conv_bn_leakyrelu_add_kernel(
    x_ptr,                    # Input tensor [N, C, H, W]
    weight_ptr,               # Conv weights [O, I, KH, KW] - for 3x3 [128, 64, 3, 3]
    running_mean_ptr,         # BN mean [O]
    running_var_ptr,          # BN var [O] 
    bn_weight_ptr,            # BN weight (gamma) [O]
    bn_bias_ptr,              # BN bias (beta) [O]
    residual_ptr,             # Residual tensor [N, O, H, W]
    output_ptr,               # Output tensor [N, O, H, W]
    n_channels_out,           # Output channels (O)
    n_channels_in,            # Input channels (I)
    n_batches,                # Batch size (N)
    height,                   # Height (H)
    width,                    # Width (W)
    kernel_size: tl.constexpr, # Kernel size (3 for 3x3)
    eps: tl.constexpr,        # BN epsilon (1e-05)
    neg_slope: tl.constexpr,  # Leaky ReLU negative slope (0.01)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)   # Batch and height dimension
    pid_n = tl.program_id(1)   # Output dimension
    
    # Calculate coordinates
    batch = pid_m // height
    h = pid_m % height
    out_c = pid_n
    
    # Load BN parameters
    bn_mean = tl.load(running_mean_ptr + out_c)
    bn_var = tl.load(running_var_ptr + out_c)
    bn_gamma = tl.load(bn_weight_ptr + out_c)
    bn_beta = tl.load(bn_bias_ptr + out_c)
    
    # Compute variance + epsilon and inverse std dev
    var_plus_eps = bn_var + eps
    inv_std_dev = tl.math.rsqrt(var_plus_eps)
    
    # Initialize output accumulator
    accumulator = tl.zeros((), tl.float16)
    
    # Convolution weights for this output channel
    weight_offset = out_c * n_channels_in * kernel_size * kernel_size
    
    # Process input channels
    for k in range(0, n_channels_in, BLOCK_SIZE_K):
        k_end = min(k + BLOCK_SIZE_K, n_channels_in)
        
        # Process spatial kernel positions
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                # For each kernel position, load corresponding input and weights
                
                # Calculate input position with padding (padding=1 for 3x3 conv)
                h_in = h - 1 + kh  # Equivalent to padding=1
                w_in = 0 + kw  # Equivalent to padding=1
                
                # Only process if we're within bounds (this effectively implements padding)
                height_valid = (h_in >= 0) and (h_in < height)
                width_valid = (w_in >= 0) and (w_in < width)
                if height_valid and width_valid:
                    # Load input slice for this spatial location
                    input_offset = batch * n_channels_in * height * width + \
                                  k * height * width + h_in * width + w_in
                    input_ptr_base = x_ptr + input_offset
                    
                    input_vals = tl.load(input_ptr_base + tl.arange(0, k_end - k), 
                                        mask=(tl.arange(0, k_end - k)) < (k_end - k),
                                        other=0.0)
                    
                    # Load weights for this output channel, input channel block, and spatial position
                    weight_offset_spatial = weight_offset + k * kernel_size * kernel_size + kh * kernel_size + kw
                    weight_ptr_base = weight_ptr + weight_offset_spatial
                    weight_vals = tl.load(weight_ptr_base + tl.arange(0, k_end - k), 
                                         mask=(tl.arange(0, k_end - k)) < (k_end - k),
                                         other=0.0)
                    
                    # Convolution operation
                    conv_res = tl.sum(input_vals * weight_vals)
                    accumulator += conv_res
    
    # Batch normalization: y = γ * (x - μ) / √(σ² + ε) + β
    normalized = (accumulator - bn_mean) * inv_std_dev
    bn_output = normalized * bn_gamma + bn_beta
    
    # Leaky ReLU: f(x) = x if x > 0, else neg_slope * x
    relu_output = tl.where(bn_output > 0, bn_output, bn_output * neg_slope)
    
    # Load residual and add
    residual_offset = batch * n_channels_out * height * width + out_c * height * width + h * width
    residual_val = tl.load(residual_ptr + residual_offset, mask=batch < n_batches and out_c < n_channels_out and h < height)
    
    final_output = relu_output + residual_val
    
    # Store result
    output_offset = batch * n_channels_out * height * width + out_c * height * width + h * width
    tl.store(output_ptr + output_offset, final_output)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_conv_bn_leakyrelu_add(x, weight, running_mean, running_var, bn_weight, bn_bias, residual):
    """Fused Conv2D + BatchNorm + LeakyReLU + Addition kernel"""
    # Get tensor shapes
    N, C, H, W = x.shape
    O, I, KH, KW = weight.shape
    
    # Verify convolution parameters from model
    assert KH == KW, "Kernel height and width must be equal"
    assert O == bn_weight.shape[0], "Output channels must match BN parameters"
    
    # For the model: conv2d(in_6, in_4, None, (1, 1), (1, 1), (1, 1), 1)
    # This means: stride=(1,1), padding=(1,1), dilation=(1,1), groups=1
    # With 3x3 kernel and padding=1, output shape should be same as input
    assert residual.shape == (N, O, H, W), "Residual shape must match output shape"
    
    # Output tensor
    output = torch.empty((N, O, H, W), device=x.device, dtype=x.dtype)
    
    # Launch kernel with appropriate parameters
    grid = (N * H, O)  # Grid dimensions: (batch_height, output_channels)
    
    fused_conv_bn_leakyrelu_add_kernel[grid](
        x_ptr=x,
        weight_ptr=weight,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        bn_weight_ptr=bn_weight,
        bn_bias_ptr=bn_bias,
        residual_ptr=residual,
        output_ptr=output,
        n_channels_out=O,
        n_channels_in=I,
        n_batches=N,
        height=H,
        width=W,
        kernel_size=KH,  # 3 for 3x3 convolution
        eps=1e-05,
        neg_slope=0.01,
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=32
    )
    
    return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_conv_bn_leakyrelu_add