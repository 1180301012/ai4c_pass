import torch
import triton
import triton.language as tl

# Pattern for flatten -> transpose -> layer_norm(128)
def pattern(ln_bias, ln_weight, conv_out):
    flattened = conv_out.flatten(2)
    transposed = flattened.transpose(1, 2)
    normalized = torch.nn.functional.layer_norm(transposed, (128,), ln_weight, ln_bias, 1e-05)
    return normalized

def replacement_args(ln_bias, ln_weight, conv_out):
    return (ln_bias, ln_weight, conv_out)

@triton.jit
def layer_norm_fused_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    B, C, HW,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one (b, hw) position
    # Input layout: [B, C, HW] accessed as [B, HW, C] after transpose
    # For position (b, hw), we need elements at x[b, :, hw] in original layout
    
    pid = tl.program_id(0)
    b = pid // HW
    hw = pid % HW
    
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < C
    
    # Load C elements strided by HW in the original [B, C, HW] layout
    # x[b, c, hw] = x_ptr + b * C * HW + c * HW + hw
    base = b * C * HW + hw
    x = tl.load(x_ptr + base + offsets * HW, mask=mask, other=0.0).to(tl.float32)
    
    C_float = C.to(tl.float32)
    mean = tl.sum(x, axis=0) / C_float
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / C_float
    std = tl.sqrt(var + eps)
    x_norm = diff / std
    
    weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = x_norm * weight + bias
    
    # Store to output [B, HW, C] layout (contiguous in C dimension)
    # out[b, hw, c] = out_ptr + b * HW * C + hw * C + c
    out_base = b * HW * C + hw * C
    tl.store(out_ptr + out_base + offsets, out, mask=mask)

@torch.fx.wrap
def fused_flatten_transpose_layernorm_128(ln_bias, ln_weight, conv_out):
    # conv_out shape: [B, C, H, W]
    B, C, H, W = conv_out.shape
    HW = H * W
    
    # Output will be [B, HW, C]
    out = torch.empty(B, HW, C, dtype=conv_out.dtype, device=conv_out.device)
    
    BLOCK_SIZE = triton.next_power_of_2(C)
    grid = (B * HW,)
    
    layer_norm_fused_kernel[grid](
        conv_out, ln_weight, ln_bias, out,
        B=B, C=C, HW=HW,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

def replacement_func():
    return fused_flatten_transpose_layernorm_128