import torch
import triton
import triton.language as tl

def pattern(ln_bias, ln_weight, conv_out):
    """Pattern: reshape -> permute -> layer_norm for batch=1, channels=64"""
    tmp_5 = conv_out.reshape(1, 64, -1)
    tmp_6 = tmp_5.permute(0, 2, 1)
    tmp_7 = torch.nn.functional.layer_norm(tmp_6, (64,), ln_weight, ln_bias, 1e-05)
    return tmp_7

def replacement_args(ln_bias, ln_weight, conv_out):
    return (ln_bias, ln_weight, conv_out)

@triton.jit
def fused_transpose_ln_kernel(
    input_ptr,   # [C, HW]
    output_ptr,  # [HW, C]
    weight_ptr,
    bias_ptr,
    HW,
    C: tl.constexpr,
    eps: tl.constexpr,
):
    # Each program handles one row of output (one HW position)
    hw_idx = tl.program_id(0)
    
    c_offs = tl.arange(0, 64)  # Use fixed size for C=64
    
    # Load column from input [C, HW] - strided access: input[c, hw_idx]
    in_offs = c_offs * HW + hw_idx
    x = tl.load(input_ptr + in_offs)
    
    # Layer norm - use float32 for better precision
    x_f32 = x.to(tl.float32)
    mean = tl.sum(x_f32, axis=0) * (1.0 / 64.0)
    x_centered = x_f32 - mean
    var = tl.sum(x_centered * x_centered, axis=0) * (1.0 / 64.0)
    rstd = 1.0 / tl.sqrt(var + eps)
    x_norm = x_centered * rstd
    
    # Affine transform
    weight = tl.load(weight_ptr + c_offs)
    bias = tl.load(bias_ptr + c_offs)
    y = x_norm * weight.to(tl.float32) + bias.to(tl.float32)
    
    # Store to output [HW, C] - contiguous write
    out_offs = hw_idx * C + c_offs
    tl.store(output_ptr + out_offs, y.to(x.dtype))

@torch.fx.wrap
def fused_reshape_permute_layernorm_b1_c64(ln_bias, ln_weight, conv_out):
    device = conv_out.device
    ln_weight = ln_weight.to(device)
    ln_bias = ln_bias.to(device)
    
    B, C, H, W = conv_out.shape
    HW = H * W
    
    # Output shape [B, HW, C]
    output = torch.empty((B, HW, C), device=device, dtype=conv_out.dtype)
    
    # Reshape input to [C, HW] for batch=1
    input_2d = conv_out.view(C, HW)
    output_2d = output.view(HW, C)
    
    # Launch kernel with more warps for better parallelism
    fused_transpose_ln_kernel[(HW,)](
        input_2d,
        output_2d,
        ln_weight,
        ln_bias,
        HW,
        C=64,
        eps=1e-5,
        num_warps=2,
    )
    
    return output

def replacement_func():
    return fused_reshape_permute_layernorm_b1_c64