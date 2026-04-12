import torch
import triton
import triton.language as tl

def pattern(x, running_mean, running_var, weight, bias):
    # BatchNorm followed by SiLU activation
    bn_output = torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    silu_output = torch.nn.functional.silu(bn_output, inplace=True)
    return silu_output

def replacement_args(x, running_mean, running_var, weight, bias):
    return (x, running_mean, running_var, weight, bias, "fused_bn_silu")

# Triton kernel for fused batch norm and SiLU
@triton.jit
def fused_bn_silu_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_channels,
    height,
    width,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    mask = idx < n_channels
    
    # Load parameters for this channel
    mean = tl.load(running_mean_ptr + idx, mask=mask, other=0.0)
    var = tl.load(running_var_ptr + idx, mask=mask, other=0.0)
    weight_val = tl.load(weight_ptr + idx, mask=mask, other=1.0)
    bias_val = tl.load(bias_ptr + idx, mask=mask, other=0.0)
    
    # Compute BatchNorm parameters
    rstd = 1.0 / tl.sqrt(var + 1e-05)
    scale = weight_val * rstd
    bias_out = bias_val - mean * scale
    
    # Process spatial dimensions
    hw_offset = tl.program_id(1) * BLOCK_SIZE_HW
    hw_idx = hw_offset + tl.arange(0, BLOCK_SIZE_HW)
    hw_mask = hw_idx < height * width
    
    # Create spatial indices
    spatial_idx = hw_idx[:, None] + idx[None, :] * height * width
    
    # Load input, compute BatchNorm, then SiLU
    x = tl.load(x_ptr + spatial_idx, mask=hw_mask[:, None], other=0.0)
    bn_out = scale[:, None] * x + bias_out[:, None]
    silu_out = bn_out * (1.0 / (1.0 + tl.exp(-bn_out)))
    
    # Store output
    out_spatial_idx = hw_idx[:, None] + idx[None, :] * height * width
    tl.store(out_ptr + out_spatial_idx, silu_out, mask=hw_mask[:, None])

# Shared replacement function for all passes
@torch.fx.wrap
def optimize_ops(*args, route=None):
    # Route based on the last argument (route string)
    if route == "fused_bn_silu":
        x, running_mean, running_var, weight, bias = args
        n_channels, height, width = x.shape[1], x.shape[2], x.shape[3]
        
        BLOCK_SIZE_C = 64
        BLOCK_SIZE_HW = 1024
        num_programs_c = (n_channels + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
        num_programs_hw = (height * width + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW
        
        out = torch.empty_like(x)
        fused_bn_silu_kernel[(num_programs_c, num_programs_hw)](
            x_ptr=x, running_mean_ptr=running_mean, running_var_ptr=running_var,
            weight_ptr=weight, bias_ptr=bias, out_ptr=out,
            n_channels=n_channels, height=height, width=width,
            BLOCK_SIZE_C=BLOCK_SIZE_C, BLOCK_SIZE_HW=BLOCK_SIZE_HW
        )
        return out
    else:
        # For other routes (not implemented yet)
        raise NotImplementedError(f"Route '{route}' not implemented")

def replacement_func():
    return optimize_ops