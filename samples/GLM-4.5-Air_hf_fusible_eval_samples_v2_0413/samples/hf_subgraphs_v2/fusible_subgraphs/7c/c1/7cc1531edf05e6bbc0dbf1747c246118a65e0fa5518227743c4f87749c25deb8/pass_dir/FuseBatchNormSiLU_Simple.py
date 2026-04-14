import torch
import triton
import triton.language as tl

def pattern(tmp_5, in_0, in_1, in_3, in_2):
    # BatchNorm computation with exact parameters from model
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    # SiLU activation (model uses inplace=True with spaces)
    tmp_7 = torch.nn.functional.silu(tmp_6, inplace = True)
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
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    n_programs = tl.cdiv(n_elements, BLOCK_SIZE)
    
    if pid >= n_programs:
        return
    
    # Calculate block bounds
    start_idx = pid * BLOCK_SIZE
    end_idx = min(start_idx + BLOCK_SIZE, n_elements)
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    
    # Load parameters they are broadcast to all spatial locations
    mean = tl.load(mean_ptr + 0)
    var = tl.load(var_ptr + 0)
    weight = tl.load(weight_ptr + 0)
    bias = tl.load(bias_ptr + 0)
    
    # Compute batch norm parameters
    std = tl.sqrt(var + 1e-05)
    inv_std = 1.0 / std
    scale = weight * inv_std
    bias_shifted = bias - mean * scale
    
    # Load input tensor for this block
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Apply batch normalization
    batch_normed = (x - mean) * scale + bias_shifted
    
    # Apply SiLU activation: x * sigmoid(x)
    sigmoid_x = 1.0 / (1.0 + tl.exp(-batch_normed))
    silu_out = batch_normed * sigmoid_x
    
    # Store output
    tl.store(out_ptr + offsets, silu_out, mask=mask)

@torch.fx.wrap
def fused_batchnorm_silu(x, mean, var, weight, bias):
    # Get tensor shape
    if len(x.shape) == 4:
        # For 4D tensor: [batch, channels, height, width]
        n_elements = x.numel()
    else:
        # For other tensor shapes
        n_elements = x.numel()
    
    # Use optimal block size
    BLOCK_SIZE = 1024
    
    # Number of programs needed
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch kernel
    fused_batchnorm_silu_kernel[(num_programs,)](
        x_ptr=x,
        mean_ptr=mean,  
        var_ptr=var,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_batchnorm_silu