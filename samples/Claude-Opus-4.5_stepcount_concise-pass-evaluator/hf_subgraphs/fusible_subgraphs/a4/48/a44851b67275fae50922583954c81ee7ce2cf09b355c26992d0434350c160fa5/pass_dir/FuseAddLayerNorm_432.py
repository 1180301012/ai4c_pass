import torch
import triton
import triton.language as tl

# Pattern for convit_small (432 dimension)
def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_2 + in_3
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (432,), in_1, in_0, 1e-06)
    return tmp_2, tmp_3

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_add_layernorm_kernel(
    in_2_ptr, in_3_ptr, weight_ptr, bias_ptr,
    out_sum_ptr, out_norm_ptr,
    M, N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (one instance of layer norm)
    row_idx = tl.program_id(0)
    row_start = row_idx * N
    
    # Load entire row
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    
    # Load inputs and compute sum - cast to float32 for precision
    in_2 = tl.load(in_2_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
    in_3 = tl.load(in_3_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
    x = in_2 + in_3
    
    # Store sum result
    tl.store(out_sum_ptr + row_start + cols, x, mask=mask)
    
    # Compute mean with masked values as 0
    x_masked = tl.where(mask, x, 0.0)
    mean = tl.sum(x_masked, axis=0) / N
    
    # Compute variance
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    
    # Compute rstd
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Load weight and bias
    weight = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Normalize and apply affine transform
    normalized = (x - mean) * rstd * weight + bias
    
    # Store normalized result
    tl.store(out_norm_ptr + row_start + cols, normalized, mask=mask)

@torch.fx.wrap
def _fused_add_layernorm_impl(in_0, in_1, in_2, in_3):
    # in_0: bias [N]
    # in_1: weight [N]
    # in_2, in_3: [B, M, N]
    
    shape = in_2.shape
    N = shape[-1]
    M = in_2.numel() // N
    
    out_sum = torch.empty_like(in_2)
    out_norm = torch.empty_like(in_2)
    
    # Choose BLOCK_SIZE as next power of 2 >= N
    BLOCK_SIZE = triton.next_power_of_2(N)
    
    # Move weight and bias to same device as input
    weight = in_1.to(in_2.device)
    bias = in_0.to(in_2.device)
    
    fused_add_layernorm_kernel[(M,)](
        in_2, in_3, weight, bias,
        out_sum, out_norm,
        M, N,
        1e-06,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out_sum, out_norm

# Unpack tuple to create separate return nodes for FX graph matching
def fused_add_layernorm(in_0, in_1, in_2, in_3):
    result = _fused_add_layernorm_impl(in_0, in_1, in_2, in_3)
    out_sum = result[0]
    out_norm = result[1]
    return out_sum, out_norm

def replacement_func():
    return fused_add_layernorm