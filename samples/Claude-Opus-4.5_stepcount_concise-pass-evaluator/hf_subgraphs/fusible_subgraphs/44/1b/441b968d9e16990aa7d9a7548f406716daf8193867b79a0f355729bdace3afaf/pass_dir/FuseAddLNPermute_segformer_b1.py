import torch
import triton
import triton.language as tl

# Pattern for add + layernorm (works for all batch sizes)
def pattern(bias, weight, x, y):
    tmp_2 = y + x
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (512,), weight, bias, 1e-05)
    return tmp_3

def replacement_args(bias, weight, x, y):
    return (bias, weight, x, y)

@triton.jit
def fused_add_ln_kernel(
    x_ptr, y_ptr, weight_ptr, bias_ptr, out_ptr,
    total_rows,
    C: tl.constexpr,
    eps: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Input offset for this row
    in_offset = pid * C
    
    # Channel offsets
    offs_c = tl.arange(0, C)
    
    # Load input data
    x = tl.load(x_ptr + in_offset + offs_c)
    y = tl.load(y_ptr + in_offset + offs_c)
    
    # Add and convert to float32 for LayerNorm
    sum_val = (x + y).to(tl.float32)
    
    # LayerNorm: compute mean
    mean = tl.sum(sum_val, axis=0) * (1.0 / C)
    
    # Compute variance
    diff = sum_val - mean
    var = tl.sum(diff * diff, axis=0) * (1.0 / C)
    
    # Normalize
    inv_std = tl.rsqrt(var + eps)
    
    # Load weight and bias and apply affine transformation
    w = tl.load(weight_ptr + offs_c).to(tl.float32)
    b = tl.load(bias_ptr + offs_c).to(tl.float32)
    normalized = diff * inv_std * w + b
    
    # Store output in original dtype
    tl.store(out_ptr + in_offset + offs_c, normalized.to(x.dtype))

@torch.fx.wrap
def fused_add_ln(bias, weight, x, y):
    # bias: [512], weight: [512]
    # x, y: [B, N, C] where C=512
    
    B = x.shape[0]
    N = x.shape[1]  # 256
    C = x.shape[2]  # 512
    total_rows = B * N
    
    device = x.device
    weight_d = weight.to(device)
    bias_d = bias.to(device)
    
    # Allocate output
    out = torch.empty_like(x)
    
    # Launch kernel with num_warps=2 - more efficient for reduction operations
    grid = (total_rows,)
    fused_add_ln_kernel[grid](
        y, x, weight_d, bias_d, out,
        total_rows,
        C=C, eps=1e-05,
        num_warps=2,
    )
    
    return out

def replacement_func():
    return fused_add_ln