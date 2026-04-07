import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_1 = in_0.flatten(1, -1)
    return (tmp_1,)

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def fused_relu_flatten_kernel(
    x_ptr,
    out_ptr,
    batch_size: tl.constexpr,
    channels: tl.constexpr,
):
    # Calculate program indices
    pid = tl.program_id(0)
    batch_offset = pid * channels
    
    # Create offset array for loading from input tensor
    offsets = batch_offset + tl.arange(0, channels)
    
    # Load input data (treating as flattened [B, C])
    x = tl.load(x_ptr + offsets, mask=offsets < batch_size * channels, other=0.0)
    
    # Apply ReLU operation
    out = tl.maximum(x, 0.0)
    
    # Store result to flattened output
    tl.store(out_ptr + offsets, out, mask=offsets < batch_size * channels)

@torch.fx.wrap
def simple_flatten(x):
    # Just flatten from dimension 1 to -1
    return x.flatten(1, -1)

def replacement_func():
    return simple_flatten