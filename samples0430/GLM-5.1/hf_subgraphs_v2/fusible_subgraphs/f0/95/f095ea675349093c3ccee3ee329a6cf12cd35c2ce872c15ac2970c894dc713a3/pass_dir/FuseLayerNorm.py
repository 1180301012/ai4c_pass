import torch
import triton
import triton.language as tl

# Simple test pass: replace layer_norm with Triton kernel (single output)
# Uses 3D strides to avoid blocked .reshape() operation

def pattern(x, weight, bias):
    return torch.nn.functional.layer_norm(x, (1024,), weight, bias, 1e-05)

def replacement_args(x, weight, bias):
    return (x, weight, bias)

# Triton kernel for layer_norm on 3D tensor (B, T, C)
@triton.jit
def layernorm_kernel_3d(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    B, T, C,
    stride_x_b, stride_x_t, stride_x_c,
    stride_out_b, stride_out_t, stride_out_c,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    time_idx = tl.program_id(1)
    
    if batch_idx >= B or time_idx >= T:
        return
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < C
    
    # Load input row: x[batch_idx, time_idx, col_offsets]
    x_ptrs = x_ptr + batch_idx * stride_x_b + time_idx * stride_x_t + col_offsets * stride_x_c
    x_vals = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    
    # Compute mean
    mean = tl.sum(x_vals, axis=0) / C
    
    # Compute variance
    diff = x_vals - mean
    var = tl.sum(diff * diff, axis=0) / C
    
    # Compute reciprocal std
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    
    # Normalize
    x_hat = (x_vals - mean) * rstd
    
    # Load weight and bias
    w_ptrs = weight_ptr + col_offsets
    b_ptrs = bias_ptr + col_offsets
    w_vals = tl.load(w_ptrs, mask=mask, other=1.0).to(tl.float32)
    b_vals = tl.load(b_ptrs, mask=mask, other=0.0).to(tl.float32)
    
    # Apply affine transform
    out_vals = x_hat * w_vals + b_vals
    
    # Store output
    out_ptrs = out_ptr + batch_idx * stride_out_b + time_idx * stride_out_t + col_offsets * stride_out_c
    tl.store(out_ptrs, out_vals, mask=mask)

@torch.fx.wrap
def triton_layernorm(x, weight, bias):
    # x shape: (B, T, C) where C = 1024
    B = x.shape[0]
    T = x.shape[1]
    C = x.shape[2]
    
    # Allocate output using allowed factory method
    out = torch.empty_like(x)
    
    # Get strides (metadata access, not ATen operation)
    stride_x_b, stride_x_t, stride_x_c = x.stride()
    stride_out_b, stride_out_t, stride_out_c = out.stride()
    
    BLOCK_SIZE = triton.next_power_of_2(C)
    
    # Grid: process each (batch, time) pair
    grid = (B, T)
    
    layernorm_kernel_3d[grid](
        x, weight, bias, out,
        B, T, C,
        stride_x_b, stride_x_t, stride_x_c,
        stride_out_b, stride_out_t, stride_out_c,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return triton_layernorm