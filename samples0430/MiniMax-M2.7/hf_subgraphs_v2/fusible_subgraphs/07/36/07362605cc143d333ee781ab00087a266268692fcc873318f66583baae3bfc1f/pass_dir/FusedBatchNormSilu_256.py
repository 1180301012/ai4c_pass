import torch
import triton
import triton.language as tl

# Pattern matching function - matches reshape -> batch_norm -> silu for 256 channels
def pattern(in_0, in_1, in_2, in_3, in_4):
    """
    Match the pattern: reshape -> batch_norm -> silu
    - in_0: running_mean
    - in_1: running_var  
    - in_2: bias (beta)
    - in_3: weight (gamma)
    - in_4: input tensor to reshape
    """
    tmp_4 = in_4.reshape(1, 256, 16, 16)
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 = torch.nn.functional.silu(tmp_5, inplace=True)
    return tmp_6

# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


@triton.jit
def fused_batchnorm_silu_kernel_256(
    input_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_elements,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and calculate offsets
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Calculate channel index
    c = (offsets // (height * width)) % channels
    
    # Load normalization parameters (broadcast across spatial dimensions)
    mean = tl.load(mean_ptr + c)
    var = tl.load(var_ptr + c)
    weight = tl.load(weight_ptr + c)
    bias = tl.load(bias_ptr + c)
    
    # Load input and compute normalized value
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # BatchNorm: (x - mean) / sqrt(var + eps) * weight + bias
    normalized = (x - mean) * weight / tl.sqrt(var + eps) + bias
    
    # SiLU: x * sigmoid(x)
    sigmoid_val = 1.0 / (1.0 + tl.exp(-normalized))
    out = normalized * sigmoid_val
    
    # Store output
    tl.store(output_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_batchnorm_silu_wrapper_256(in_0, in_1, in_2, in_3, in_4):
    """
    Fused kernel for reshape -> batch_norm -> silu (256 channels, 16x16 spatial)
    Input shape: (batch, in_channels, H*W) -> (1, channels, H, W) after reshape
    """
    batch = in_4.shape[0]
    channels = in_4.shape[1]
    hw = in_4.shape[2]
    
    # Calculate spatial dimensions (assuming square spatial)
    spatial_size = int(hw ** 0.5)
    height = spatial_size
    width = spatial_size
    
    n_elements = batch * channels * height * width
    
    # Reshape input to (batch, channels, height, width)
    input_reshaped = in_4.reshape(batch, channels, height, width)
    
    # Contiguous view for kernel access: (batch, height, width, channels)
    input_flat = input_reshaped.permute(0, 2, 3, 1).reshape(-1)
    input_flat = input_flat.reshape(batch * height * width, channels)
    
    # Create output tensor with same layout
    output = torch.empty_like(in_4).reshape(batch, channels, height, width).permute(0, 2, 3, 1).reshape(-1)
    
    # Expand parameters to match output layout: (batch*h*w, channels)
    mean = in_0.reshape(1, 1, 1, channels).expand(batch, height, width, channels).reshape(-1)
    var = in_1.reshape(1, 1, 1, channels).expand(batch, height, width, channels).reshape(-1)
    weight = in_3.reshape(1, 1, 1, channels).expand(batch, height, width, channels).reshape(-1)
    bias = in_2.reshape(1, 1, 1, channels).expand(batch, height, width, channels).reshape(-1)
    
    BLOCK_SIZE = 1024
    num_programs = (batch * channels * height * width + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_batchnorm_silu_kernel_256[(num_programs,)](
        input_ptr=input_flat,
        mean_ptr=mean,
        var_ptr=var,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        n_elements=n_elements,
        channels=channels,
        height=height,
        width=width,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape output back to original format
    output = output.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
    return output.reshape(batch, channels, hw)


def replacement_func():
    return fused_batchnorm_silu_wrapper_256