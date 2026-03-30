import torch
import triton
import triton.language as tl

def pattern(in_4, in_0, in_1, in_3, in_2):
    # Reshape input to 4D batch format
    tmp_4 = in_4.reshape(1, -1, in_4.shape[1], in_4.shape[2])
    
    # Batch normalization
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    
    # SiLU activation
    tmp_6 = torch.nn.functional.silu(tmp_5, inplace=True)
    
    return (tmp_6,)

def replacement_args(in_4, in_0, in_1, in_3, in_2):
    return (in_4, in_0, in_1, in_3, in_2)

@triton.jit
def fused_batch_norm_silu_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    num_channels,
    eps: tl.constexpr,
    momentum: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles a block of elements
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate spatial indices for loading parameters
    channel_idx = (offsets // (x_ptr.shape[2] * x_ptr.shape[3])) % num_channels
    
    # Load batch norm parameters - one value per channel
    mean = tl.load(running_mean_ptr + channel_idx, mask=channel_idx < num_channels, other=0.0)
    var = tl.load(running_var_ptr + channel_idx, mask=channel_idx < num_channels, other=1.0)
    weight = tl.load(weight_ptr + channel_idx, mask=channel_idx < num_channels, other=1.0)
    bias = tl.load(bias_ptr + channel_idx, mask=channel_idx < num_channels, other=0.0)
    
    # Perform batch normalization
    var_inv = 1.0 / tl.sqrt(var + eps)
    x_norm = (x - mean) * var_inv
    y = x_norm * weight + bias
    
    # Apply SiLU activation: y = x * sigmoid(x)
    y_silu = y * tl.sigmoid(y)
    
    # Store result
    tl.store(out_ptr + offsets, y_silu, mask=mask)

@torch.fx.wrap
def fused_batch_norm_silu(in_4, running_mean, running_var, weight, bias):
    # Get input tensor info
    N, C_in, H, W = 1, in_4.shape[0], in_4.shape[1], in_4.shape[2]
    C_out = running_mean.shape[0]
    
    # Determine spatial dimensions after reshaping
    if C_in == C_out:
        # Case: [4, C, H] -> [1, C, H, H] (assuming square spatial dims)
        spatial_dim = H
        x_reshaped_shape = (1, C_out, spatial_dim, spatial_dim)
        n_elements = C_out * spatial_dim * spatial_dim
    else:
        # Case where channel numbers differ, need to infer spatial dims
        total_elements = in_4.numel()
        if total_elements % C_out == 0:
            spatial_elements = total_elements // C_out
            # Assuming square spatial dimensions
            spatial_dim = int(spatial_elements ** 0.5)
            x_reshaped_shape = (1, C_out, spatial_dim, spatial_dim)
            n_elements = total_elements
        else:
            # Fallback to linear reshape
            x_reshaped_shape = (1, C_out, -1)
            n_elements = total_elements
    
    # Create output tensor
    out = torch.empty(x_reshaped_shape, dtype=in_4.dtype, device=in_4.device)
    
    # Choose block size
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_batch_norm_silu_kernel[(num_programs,)](
        x_ptr=in_4,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        n_elements=n_elements,
        num_channels=C_out,
        eps=1e-05,
        momentum=0.1,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return fused_batch_norm_silu