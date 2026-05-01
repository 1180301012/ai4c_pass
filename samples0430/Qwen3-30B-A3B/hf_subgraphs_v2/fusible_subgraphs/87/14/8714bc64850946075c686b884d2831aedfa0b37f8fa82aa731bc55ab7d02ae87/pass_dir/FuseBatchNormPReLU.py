import torch
import triton
import triton.language as tl

# Pattern matching function

def pattern(tmp_7, tmp_1, tmp_2, tmp_4, tmp_3, tmp_0):
    batchnorm_output = torch.nn.functional.batch_norm(tmp_7, tmp_1, tmp_2, tmp_4, tmp_3, False, 0.1, 0.001)
    prelu_output = torch.prelu(batchnorm_output, tmp_0)
    return prelu_output

# Argument extraction function

def replacement_args(tmp_7, tmp_1, tmp_2, tmp_4, tmp_3, tmp_0):
    return (tmp_7, tmp_1, tmp_2, tmp_4, tmp_3, tmp_0)

# Triton kernel implementation
@triton.jit
def batchnorm_prelu_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    slope_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Calculate global thread index
    pid = tl.program_id(0)
    
    # Calculate batch, height, and width indices
    b = pid // (height * width)
    h = (pid % (height * width)) // width
    w = pid % width
    
    # Calculate start index for this spatial location
    start_idx = b * channels * height * width + h * width * channels + w * channels
    
    # Load input data for the spatial location
    x = tl.load(x_ptr + start_idx, channels)
    
    # Load channel-specific parameters
    running_means = tl.load(running_mean_ptr, channels)
    running_vars = tl.load(running_var_ptr, channels)
    weights = tl.load(weight_ptr, channels)
    biases = tl.load(bias_ptr, channels)
    slopes = tl.load(slope_ptr, channels)
    
    # Process all channels for this spatial location
    for c in range(channels):
        # Normalize
        x_hat = (x[c] - running_means[c]) / tl.sqrt(running_vars[c] + eps)
        
        # Batch norm
        y = weights[c] * x_hat + biases[c]
        
        # PReLU
        out_val = y if y >= 0 else slopes[c] * y
        
        # Store result
        tl.store(out_ptr + start_idx + c, out_val)

# Kernel wrapper
@torch.fx.wrap
def fused_batchnorm_prelu(x, running_mean, running_var, weight, bias, slope):
    batch_size, channels, height, width = x.shape
    out = torch.empty_like(x)
    
    # Define block size (can be optimized for performance)
    BLOCK_SIZE = 1024
    num_blocks = (batch_size * height * width + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the kernel
    batchnorm_prelu_kernel[(num_blocks,)](
        x_ptr=x,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        slope_ptr=slope,
        out_ptr=out,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        eps=0.001,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

# Replacement function

def replacement_func():
    return fused_batchnorm_prelu