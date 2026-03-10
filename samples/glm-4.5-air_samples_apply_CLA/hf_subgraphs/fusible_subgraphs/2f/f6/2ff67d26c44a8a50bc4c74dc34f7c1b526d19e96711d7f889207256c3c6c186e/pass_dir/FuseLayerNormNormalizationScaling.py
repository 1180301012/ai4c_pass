import torch
import triton
import triton.language as tl
import math

def pattern(in_0, in_1, in_2, in_3):
    # Match the complete LayerNorm computation pattern
    tmp_3 = in_3 + in_2
    tmp_4 = tmp_3.float()
    tmp_5 = tmp_4.mean(-1, keepdim=True)
    tmp_6 = tmp_4 - tmp_5
    tmp_7 = tmp_6.pow(2)
    tmp_8 = tmp_7.mean(-1, keepdim=True)
    tmp_9 = tmp_4 - tmp_5
    tmp_10 = tmp_8 + 1e-07
    tmp_11 = torch.sqrt(tmp_10)
    tmp_12 = tmp_9 / tmp_11
    tmp_13 = tmp_12.to(torch.float32)
    tmp_14 = in_1 * tmp_13
    tmp_15 = tmp_14 + in_0
    return tmp_15

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_layernorm_kernel(
    bias_ptr,
    weight_ptr,
    addend_ptr,
    input_ptr,
    out_ptr,
    batch_seq_size,
    n_features: tl.constexpr,
    EPS: tl.constexpr,
):
    # Each program handles one [batch, seq] combination
    pid = tl.program_id(0)
    
    if pid >= batch_seq_size:
        return
    
    # Starting offset for this [batch, seq] combination
    base_offset = pid * n_features
    
    # Use tl.arange with power of 2 block size - fixed approach
    offsets = tl.arange(0, 256)  # Fixed 256 elements (power of 2)
    
    # Mask for valid elements
    mask = offsets < n_features
    
    # Load multiple elements at once for vectorization
    x = tl.load(input_ptr + base_offset, mask=mask, other=0.0).to(tl.float32)
    addend = tl.load(addend_ptr + base_offset, mask=mask, other=0.0).to(tl.float32)
    
    x_total = x + addend
    
    # Compute mean using vectorized operations
    active_count = tl.sum(mask.to(tl.float32))
    sum_val = tl.sum(x_total)
    mean = sum_val / active_count
    
    # Compute variance
    x_centered = x_total - mean
    sum_sq = tl.sum(x_centered * x_centered)
    var = sum_sq / active_count
    var = max(var, EPS)
    std = tl.sqrt(var)
    
    # Load weight and bias
    weight = tl.load(weight_ptr, mask=mask, other=1.0)
    bias = tl.load(bias_ptr, mask=mask, other=0.0)
    
    # Apply LayerNorm transformation
    x_norm = x_centered / std
    out = x_norm * weight + bias
    
    # Store output
    tl.store(out_ptr + base_offset, out, mask=mask)

@torch.fx.wrap
def fused_layernorm(in_0, in_1, in_2, in_3):
    # in_0: bias tensor
    # in_1: weight tensor  
    # in_2: addend tensor
    # in_3: input tensor
    
    # Get input tensor info
    input_shape = in_3.shape
    batch_size = input_shape[0]
    seq_len = input_shape[1] if len(input_shape) > 1 else 1
    n_features = input_shape[-1]  # 768
    
    # Total number of [batch, seq] combinations to process
    batch_seq_size = batch_size * seq_len
    
    # Output tensor
    out = torch.empty_like(in_3)
    
    # Simple 2D grid: [batch_seq_size, n_features]
    fused_layernorm_kernel[(batch_seq_size, n_features)](
        in_0,
        in_1,
        in_2,
        in_3,
        out,
        batch_seq_size,
        n_features,
        1e-07
    )
    
    return out

def replacement_func():
    return fused_layernorm