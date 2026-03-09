import torch
import triton
import triton.language as tl

def pattern(tmp_5, tmp_1, tmp_2, tmp_4, tmp_3, tmp_0):
    # Match BatchNorm + PReLU sequence
    # tmp_5 is the input tensor to batch norm (concatenated result)
    # tmp_1, tmp_2, tmp_4, tmp_3 are batch norm parameters  
    # tmp_0 is the PReLU weight
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, tmp_1, tmp_2, tmp_4, tmp_3, False, 0.1, 0.001)
    tmp_7 = torch.prelu(tmp_6, tmp_0)
    return tmp_6, tmp_7

def replacement_args(tmp_5, tmp_1, tmp_2, tmp_4, tmp_3, tmp_0):
    return (tmp_5, tmp_1, tmp_2, tmp_4, tmp_3, tmp_0)

@triton.jit
def fused_batchnorm_prelu_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    prelu_weight_ptr,
    out_ptr,
    n_channels,
    height,
    width,
    momentum: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CHANNEL_BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one channel
    channel_idx = tl.program_id(0)
    chan_start = channel_idx * CHANNEL_BLOCK_SIZE
    chan_offsets = chan_start + tl.arange(0, CHANNEL_BLOCK_SIZE)
    chan_mask = chan_offsets < n_channels
    
    # Compute spatial offsets
    spatial_size = height * width
    spatial_offsets = tl.arange(0, BLOCK_SIZE)
    mask = spatial_offsets < spatial_size
    
    # Load batch norm parameters for this channel
    mean = tl.load(running_mean_ptr + chan_offsets, mask=chan_mask, other=0.0).to(tl.float32)
    var = tl.load(running_var_ptr + chan_offsets, mask=chan_mask, other=1.0).to(tl.float32)
    weight = tl.load(weight_ptr + chan_offsets, mask=chan_mask, other=1.0).to(tl.float32)
    bias = tl.load(bias_ptr + chan_offsets, mask=chan_mask, other=0.0).to(tl.float32)
    prelu_weight = tl.load(prelu_weight_ptr + chan_offsets, mask=chan_mask, other=0.0).to(tl.float32)
    
    # Normalize var to get inv std
    inv_std = tl.rsqrt(var + eps)
    
    # Load input data for this spatial block
    x_ptrs = x_ptr + (chan_offsets[:, None] * spatial_size + spatial_offsets[None, :])
    x = tl.load(x_ptrs, mask=chan_mask[:, None] & mask[None, :], other=0.0).to(tl.float32)
    
    # Apply batch normalization
    norm = (x - mean[:, None]) * inv_std[:, None]
    bn_out = norm * weight[:, None] + bias[:, None]
    
    # Apply PReLU activation
    out = tl.where(bn_out > 0, bn_out, bn_out * prelu_weight[:, None])
    
    # Store result
    out_ptrs = out_ptr + (chan_offsets[:, None] * spatial_size + spatial_offsets[None, :])
    tl.store(out_ptrs, out, mask=chan_mask[:, None] & mask[None, :])

@torch.fx.wrap
def fused_batchnorm_prelu(x, running_mean, running_var, weight, bias, prelu_weight):
    n_channels, height, width = x.shape[1], x.shape[2], x.shape[3]
    spatial_size = height * width
    n_elements = n_channels * spatial_size
    
    # Tune block sizes for optimal performance
    BLOCK_SIZE = 1024  # Spatial block size
    CHANNEL_BLOCK_SIZE = 128  # Channels per program
    
    # Calculate grid dimensions
    num_channel_programs = (n_channels + CHANNEL_BLOCK_SIZE - 1) // CHANNEL_BLOCK_SIZE
    num_spatial_programs = (spatial_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Output tensor
    out = torch.empty_like(x)
    
    # Launch kernel
    fused_batchnorm_prelu_kernel[
        (num_channel_programs, num_spatial_programs)
    ](
        x_ptr=x,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        prelu_weight_ptr=prelu_weight,
        out_ptr=out,
        n_channels=n_channels,
        height=height,
        width=width,
        momentum=0.1,
        eps=0.001,
        BLOCK_SIZE=BLOCK_SIZE,
        CHANNEL_BLOCK_SIZE=CHANNEL_BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_batchnorm_prelu