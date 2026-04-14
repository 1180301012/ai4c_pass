import torch
import triton
import triton.language as tl

def pattern(tmp_5, in_0, in_1, in_3, in_2):
    # BatchNorm computation
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    # SiLU activation (swish: x * sigmoid(x))
    tmp_7 = torch.nn.functional.silu(tmp_6, inplace=True)
    return tmp_7

def replacement_args(tmp_5, in_0, in_1, in_3, in_2):
    return (tmp_5, in_0, in_1, in_3, in_2)

@triton.jit
def fused_batchnorm_silu_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_channels,
    height,
    width,
    eps: tl.constexpr,
    momentum: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Each program handles a block of channels
    channel_idx = tl.program_id(0)
    
    # Check if this program is within bounds
    if channel_idx >= n_channels:
        return
    
    # Load parameters for this channel (from CPU, these are small)
    mean = tl.load(mean_ptr + channel_idx)
    var = tl.load(var_ptr + channel_idx)
    weight = tl.load(weight_ptr + channel_idx)
    bias = tl.load(bias_ptr + channel_idx)
    
    # Compute batch norm parameters
    std = tl.sqrt(var + eps)
    inv_std = 1.0 / std
    scale = weight * inv_std
    bias_shifted = bias - mean * scale
    
    # Process each spatial position
    for h_idx in range(0, height, BLOCK_SIZE_M):
        for w_idx in range(0, width, BLOCK_SIZE_N):
            # Compute current block bounds
            h_end = min(h_idx + BLOCK_SIZE_M, height)
            w_end = min(w_idx + BLOCK_SIZE_N, width)
            
            # Load input data for this spatial block
            x_ptr_base = x_ptr + channel_idx * height * width + h_idx * width + w_idx
            x_vals = tl.load(x_ptr_base + tl.arange(0, BLOCK_SIZE_M * BLOCK_SIZE_N), 
                           mask=(tl.arange(0, BLOCK_SIZE_M * BLOCK_SIZE_N) + h_idx * width + w_idx < height * width),
                           other=0.0).to(tl.float32)
            
            # Reshape to 2D for spatial processing
            x_2d = x_vals.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N)
            
            # Batch norm computation
            batch_normed = (x_2d - mean) * scale + bias_shifted
            
            # SiLU activation: x * sigmoid(x)
            sigmoid_x = 1.0 / (1.0 + tl.exp(-batch_normed))
            silu_out = batch_normed * sigmoid_x
            
            # Store output
            out_ptr_base = out_ptr + channel_idx * height * width + h_idx * width + w_idx
            silu_flat = silu_out.reshape(BLOCK_SIZE_M * BLOCK_SIZE_N)
            store_mask = (tl.arange(0, BLOCK_SIZE_M * BLOCK_SIZE_N) + h_idx * width + w_idx < height * width)
            tl.store(out_ptr_base, silu_flat, mask=store_mask)

@torch.fx.wrap
def fused_batchnorm_silu(x, mean, var, weight, bias):
    # Get tensor shapes
    n_channels = x.size(1)
    height = x.size(2)
    width = x.size(3)
    n_elements = n_channels * height * width
    
    # Use appropriate block sizes based on tensor dimensions
    BLOCK_SIZE_M = 16  # Block height
    BLOCK_SIZE_N = 16  # Block width
    
    # Number of programs needed (one per channel)
    n_programs = n_channels
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch kernel
    fused_batchnorm_silu_kernel[(n_programs, 1, 1)](
        x_ptr=x,
        mean_ptr=mean,
        var_ptr=var,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        n_channels=n_channels,
        height=height,
        width=width,
        eps=1e-05,
        momentum=0.1,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return out

def replacement_func():
    return fused_batchnorm_silu