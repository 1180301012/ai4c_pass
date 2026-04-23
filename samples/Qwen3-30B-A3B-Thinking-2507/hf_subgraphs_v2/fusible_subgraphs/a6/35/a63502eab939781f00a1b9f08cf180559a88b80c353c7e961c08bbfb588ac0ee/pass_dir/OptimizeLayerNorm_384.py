import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(x, weight, bias):
    return torch.nn.functional.layer_norm(x, (384,), weight, bias, 1e-05)

# Argument extraction function

def replacement_args(x, weight, bias):
    return (x, weight, bias)

# Triton layer norm kernel
@triton.jit
def layer_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    L, H, eps,
    BLOCK_SIZE_H: tl.constexpr,
):
    x_idx = tl.program_id(0)  # sequence index (0 to L-1)
    
    start_H = tl.thread_idx(0) * BLOCK_SIZE_H
    end_H = start_H + BLOCK_SIZE_H
    H_block = tl.min(BLOCK_SIZE_H, H - start_H)
    
    # Load x as float16/bfloat16 and convert to float32
    x = tl.load(x_ptr + x_idx * H + start_H, mask=start_H + tl.arange(0, H_block) < H)
    x = x.to(tl.float32)
    
    # Partial sums in float32
    sum1 = tl.zeros(H_block, dtype=tl.float32)
    sum2 = tl.zeros(H_block, dtype=tl.float32)
    sum1 += x
    sum2 += x * x
    
    # Total sums
    sum1_total = tl.sum(sum1)
    sum2_total = tl.sum(sum2)
    
    # Compute mean and variance (float32)
    mean = sum1_total / H
    var = (sum2_total / H) - (mean * mean)
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    # Load weight and bias as float32
    weight = tl.load(weight_ptr + start_H, mask=start_H + tl.arange(0, H_block) < H)
    weight = weight.to(tl.float32)
    bias = tl.load(bias_ptr + start_H, mask=start_H + tl.arange(0, H_block) < H)
    bias = bias.to(tl.float32)
    
    # Compute output
    normalized = (x - mean) * inv_std
    out = normalized * weight + bias
    out = out.to(x.dtype)  # Cast back to input dtype
    
    # Store
    tl.store(out_ptr + x_idx * H + start_H, out, mask=start_H + tl.arange(0, H_block) < H)

# Kernel wrapper
@torch.fx.wrap
def layer_norm_wrapper(x, weight, bias):
    B, L, H = x.shape
    
    # Handle batch size (all examples have B=1)
    if B != 1:
        raise ValueError(f"Batch size must be 1, got {B}")

    out = torch.empty_like(x)
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    # Configure kernel parameters
    BLOCK_SIZE_H = 256
    
    # Launch kernel: one block per sequence element
    layer_norm_kernel[(L,)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        L=L,
        H=H,
        eps=1e-05,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
    )
    
    return out

# Replacement function

def replacement_func():
    return layer_norm_wrapper