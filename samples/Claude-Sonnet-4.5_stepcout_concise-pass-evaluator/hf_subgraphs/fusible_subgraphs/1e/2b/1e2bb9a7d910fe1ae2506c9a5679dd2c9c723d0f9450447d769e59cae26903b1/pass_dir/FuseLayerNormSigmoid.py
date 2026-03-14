import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_3, weight, bias):
    """ 
    Match layer_norm followed by sigmoid
    """
    tmp_2 = torch.nn.functional.layer_norm(in_3, (256,), weight, bias, 1e-05)
    tmp_4 = tmp_2.sigmoid()
    return tmp_4

# Argument extraction function
def replacement_args(in_3, weight, bias):
    return (in_3, weight, bias)

# Simple, efficient fused kernel
@triton.jit
def fused_layernorm_sigmoid_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    M,  # number of rows
    N: tl.constexpr,  # feature dimension (256)
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    row_offset = row_idx * N
    
    # Load entire row
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load all data
    x = tl.load(input_ptr + row_offset + offsets, mask=mask, other=0.0)
    w = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
    b = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    
    # Compute mean and variance
    mean = tl.sum(tl.where(mask, x, 0.0)) / N
    x_centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_centered * x_centered) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Apply layer norm and sigmoid in one shot
    out = tl.sigmoid((x_centered * rstd) * w + b)
    
    # Store
    tl.store(output_ptr + row_offset + offsets, out, mask=mask)

# Kernel wrapper
@torch.fx.wrap
def fused_layernorm_sigmoid(input_tensor, weight, bias):
    orig_shape = input_tensor.shape
    input_2d = input_tensor.reshape(-1, 256).contiguous()
    M = input_2d.shape[0]
    
    output = torch.empty_like(input_2d)
    
    # Launch with optimal settings for this size
    grid = (M,)
    fused_layernorm_sigmoid_kernel[grid](
        input_2d, weight.contiguous(), bias.contiguous(), output,
        M=M, N=256, eps=1e-05, BLOCK_SIZE=256,
        num_warps=4, num_stages=1,
    )
    
    return output.reshape(orig_shape)

# Replacement function
def replacement_func():
    return fused_layernorm_sigmoid