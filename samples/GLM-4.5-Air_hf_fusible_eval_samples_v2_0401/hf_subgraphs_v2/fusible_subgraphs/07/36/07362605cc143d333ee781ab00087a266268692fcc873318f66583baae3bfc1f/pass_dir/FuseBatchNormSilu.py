import torch
import triton
import triton.language as tl

def pattern(in_4, in_0, in_1, in_2, in_3):
    """Match reshape + batch_norm + silu pattern"""
    tmp_4 = in_4.reshape(1, 512, 8, 8)
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 = torch.nn.functional.silu(tmp_5, inplace=True)
    return (tmp_6,)

def replacement_args(in_4, in_0, in_1, in_2, in_3):
    """Extract arguments for the fused kernel"""
    return (in_4, in_0, in_1, in_2, in_3)

@triton.jit
def fused_batchnorm_silu_kernel(
    x_ptr,           # Input tensor [batch, channels, height, width]
    mean_ptr,        # Running mean [channels]
    var_ptr,         # Running var [channels] 
    weight_ptr,      # Weight [channels]
    bias_ptr,        # Bias [channels]
    out_ptr,         # Output tensor [batch, channels, height, width]
    n_channels,
    n_elements,      # Total elements in the tensor
    momentum,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data with proper broadcasting
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load batch norm parameters
    channel_idx = offsets % n_channels
    
    # Load parameters for each element's channel
    mean = tl.load(mean_ptr + channel_idx, mask=channel_idx < n_channels, other=0.0)
    var = tl.load(var_ptr + channel_idx, mask=channel_idx < n_channels, other=1.0)
    weight = tl.load(weight_ptr + channel_idx, mask=channel_idx < n_channels, other=1.0)
    bias = tl.load(bias_ptr + channel_idx, mask=channel_idx < n_channels, other=0.0)
    
    # Batch normalization: (x - mean) / sqrt(var + eps) * weight + bias
    denom = tl.sqrt(var + eps)
    normalized = (x - mean) / denom
    batchnorm_out = normalized * weight + bias
    
    # SiLU activation: x * sigmoid(x)
    silu_out = batchnorm_out * tl.sigmoid(batchnorm_out)
    
    # Store result
    tl.store(out_ptr + offsets, silu_out, mask=mask)

@torch.fx.wrap
def fused_batchnorm_silu(in_4, in_0, in_1, in_2, in_3):
    """Fused batch norm + silu kernel wrapper"""
    # Get input dimensions dynamically
    original_shape = in_4.shape
    batch, channels, height, width = 1, original_shape[1] * original_shape[0], original_shape[2], 8
    
    # Determine spatial dimensions from input shape
    if max(height, width) > 1:
        # Input is [4, C, H] where H needs to be split into spatial dims
        total_spatial = original_shape[2]
        if total_spatial % 8 == 0:
            height = total_spatial // 8
            width = 8
        else:
            height = 8
            width = total_spatial // 8
    
    n_elements = batch * channels * height * width
    
    # Determine target channel count from running_mean/var
    n_channels = in_0.shape[0] if len(in_0.shape) > 0 else 512
    
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(in_4)
    
    # Reshape input to [batch, channels, height, width] layout
    # The input is already in the right shape format, just need to flatten properly
    if len(in_4.shape) == 3:  # [4, C, H] case
        # Reshape to [1, 4*C, H//8, 8] or [1, 4*C, 8, H//8]
        if in_4.shape[2] % 8 == 0:
            reshaped = in_4.reshape(1, in_4.shape[0] * in_4.shape[1], in_4.shape[2] // 8, 8)
        else:
            reshaped = in_4.reshape(1, in_4.shape[0] * in_4.shape[1], 8, in_4.shape[2] // 8)
    else:  # Already has proper shape
        reshaped = in_4
    
    # Launch kernel
    fused_batchnorm_silu_kernel[(num_programs,)](
        reshaped,
        in_0,      # running_mean
        in_1,      # running_var
        in_2,      # weight
        in_3,      # bias
        out,
        n_channels,
        n_elements,
        0.1,       # momentum
        1e-05,     # eps
        BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    """Return fused kernel function"""
    return fused_batchnorm_silu