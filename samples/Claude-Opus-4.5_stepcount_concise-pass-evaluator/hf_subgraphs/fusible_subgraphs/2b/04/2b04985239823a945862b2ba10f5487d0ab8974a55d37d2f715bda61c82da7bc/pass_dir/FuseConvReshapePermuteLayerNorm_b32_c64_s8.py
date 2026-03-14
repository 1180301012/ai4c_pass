import torch
import triton
import triton.language as tl

def pattern(ln_bias, ln_weight, conv_out):
    """Pattern: reshape -> permute -> layer_norm for batch=32, channels=64"""
    tmp_5 = conv_out.reshape(32, 64, -1)
    tmp_6 = tmp_5.permute(0, 2, 1)
    tmp_7 = torch.nn.functional.layer_norm(tmp_6, (64,), ln_weight, ln_bias, 1e-05)
    return tmp_7

def replacement_args(ln_bias, ln_weight, conv_out):
    return (ln_bias, ln_weight, conv_out)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['C'],
)
@triton.jit
def fused_permute_layernorm_kernel_b32_c64(
    input_ptr,   # Input tensor [B, C, HW]
    output_ptr,  # Output tensor [B, HW, C]
    weight_ptr,  # Layer norm weight [C]
    bias_ptr,    # Layer norm bias [C]
    B,           # Batch size
    C: tl.constexpr,  # Channel count
    HW,          # H * W
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    batch_idx = pid // HW
    hw_idx = pid % HW
    
    # Load the C values for this (batch, hw) position
    c_offsets = tl.arange(0, BLOCK_SIZE)
    mask = c_offsets < C
    
    # Input offset = batch_idx * (C * HW) + c * HW + hw_idx
    input_offsets = batch_idx * C * HW + c_offsets * HW + hw_idx
    x = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
    
    # Compute mean
    mean = tl.sum(x, axis=0) / C
    
    # Compute variance
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / C
    
    # Normalize
    rstd = 1.0 / tl.sqrt(var + eps)
    x_norm = x_centered * rstd
    
    # Load weight and bias
    weight = tl.load(weight_ptr + c_offsets, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + c_offsets, mask=mask, other=0.0)
    
    # Apply affine transform
    y = x_norm * weight + bias
    
    # Store result - output is [B, HW, C]
    output_offsets = batch_idx * HW * C + hw_idx * C + c_offsets
    tl.store(output_ptr + output_offsets, y, mask=mask)

@torch.fx.wrap
def fused_reshape_permute_layernorm_b32_c64(ln_bias, ln_weight, conv_out):
    device = conv_out.device
    ln_weight = ln_weight.to(device)
    ln_bias = ln_bias.to(device)
    
    B, C, H, W = conv_out.shape
    HW = H * W
    
    # Allocate output
    output = torch.empty((B, HW, C), device=device, dtype=conv_out.dtype)
    
    # Make input contiguous and reshape to [B, C, HW]
    conv_flat = conv_out.contiguous().view(B, C, HW)
    
    # Launch kernel
    grid = (B * HW,)
    fused_permute_layernorm_kernel_b32_c64[grid](
        conv_flat,
        output,
        ln_weight,
        ln_bias,
        B, C, HW,
        eps=1e-5,
    )
    
    return output

def replacement_func():
    return fused_reshape_permute_layernorm_b32_c64