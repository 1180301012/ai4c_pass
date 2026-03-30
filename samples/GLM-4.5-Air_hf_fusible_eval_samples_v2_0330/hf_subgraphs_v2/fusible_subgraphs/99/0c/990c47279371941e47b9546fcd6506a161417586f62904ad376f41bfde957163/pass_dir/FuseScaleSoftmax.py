import torch
import triton
import triton.language as tl
import math

def pattern(in_0):
    tmp_0 = 0.0625 * in_0
    tmp_1 = torch.nn.functional.softmax(tmp_0, dim=-1)
    return tmp_1

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def fused_scale_softmax_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    seq_len,
    num_heads,
    scale_factor: tl.constexpr,
):
    # Each program processes one head in one batch
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    # Calculate the offset for this batch and head
    batch_stride = seq_len * num_heads
    head_stride = seq_len
    offset = batch_idx * batch_stride + head_idx * head_stride
    
    # Load the entire sequence for this head in float for better precision
    x_seq = tl.load(x_ptr + offset, mask=tl.arange(0, seq_len) < seq_len, other=float('-inf'))
    
    # Apply scaling
    x_scaled = scale_factor * x_seq
    
    # Compute max for numerical stability
    max_val = tl.max(x_scaled)
    
    # Compute exp and sum
    exp_vals = tl.exp(x_scaled - max_val)
    sum_exp = tl.sum(exp_vals)
    
    # Compute softmax
    softmax_vals = exp_vals / sum_exp
    
    # Store the result
    tl.store(out_ptr + offset, softmax_vals, mask=tl.arange(0, seq_len) < seq_len)

@torch.fx.wrap
def fused_scale_softmax(x):
    batch_size, seq_len, num_heads = x.shape
    scale_factor = 0.0625
    
    output = torch.empty_like(x)
    
    # Launch kernel with correct grid dimensions
    fused_scale_softmax_kernel[(batch_size, num_heads, 1)](
        x_ptr=x,
        out_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=num_heads,
        scale_factor=scale_factor,
    )
    
    return output

def replacement_func():
    return fused_scale_softmax