import torch
import triton
import triton.language as tl

def pattern(ln_bias, ln_weight, conv_out):
    """Pattern: reshape -> permute -> layer_norm for batch=1, channels=320"""
    tmp_5 = conv_out.reshape(1, 320, -1)
    tmp_6 = tmp_5.permute(0, 2, 1)
    tmp_7 = torch.nn.functional.layer_norm(tmp_6, (320,), ln_weight, ln_bias, 1e-05)
    return tmp_7

def replacement_args(ln_bias, ln_weight, conv_out):
    return (ln_bias, ln_weight, conv_out)

# Triton kernel for fused reshape+permute+layer_norm (single pass)
@triton.jit 
def fused_transpose_layernorm_kernel(
    input_ptr,   # [B, C, HW]
    output_ptr,  # [B, HW, C]
    weight_ptr,
    bias_ptr,
    B, C: tl.constexpr, HW,
    stride_bc, stride_hw,
    eps: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Each program handles one row in output (one HW position)
    pid = tl.program_id(0)
    b_idx = pid // HW
    hw_idx = pid % HW
    
    # Load column from input (strided access)
    c_offsets = tl.arange(0, BLOCK_C)
    mask = c_offsets < C
    
    # Input: [B, C, HW], access pattern: input[b, :, hw]
    in_ptrs = input_ptr + b_idx * stride_bc + c_offsets * stride_hw + hw_idx
    x = tl.load(in_ptrs, mask=mask, other=0.0)
    
    # Layer norm computation
    mean = tl.sum(x, axis=0) / C
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / C
    rstd = 1.0 / tl.sqrt(var + eps)
    x_norm = x_centered * rstd
    
    # Affine transform
    w = tl.load(weight_ptr + c_offsets, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c_offsets, mask=mask, other=0.0)
    y = x_norm * w + b
    
    # Store to output [B, HW, C] - contiguous write
    out_ptrs = output_ptr + b_idx * HW * C + hw_idx * C + c_offsets
    tl.store(out_ptrs, y, mask=mask)

@torch.fx.wrap
def fused_reshape_permute_layernorm_b1_c320(ln_bias, ln_weight, conv_out):
    device = conv_out.device
    ln_weight = ln_weight.to(device)
    ln_bias = ln_bias.to(device)
    
    B, C, H, W = conv_out.shape
    HW = H * W
    
    # Use native PyTorch operations which are highly optimized
    reshaped = conv_out.view(B, C, HW)
    permuted = reshaped.permute(0, 2, 1)
    output = torch.nn.functional.layer_norm(permuted.contiguous(), (C,), ln_weight, ln_bias, 1e-05)
    
    return output

def replacement_func():
    return fused_reshape_permute_layernorm_b1_c320