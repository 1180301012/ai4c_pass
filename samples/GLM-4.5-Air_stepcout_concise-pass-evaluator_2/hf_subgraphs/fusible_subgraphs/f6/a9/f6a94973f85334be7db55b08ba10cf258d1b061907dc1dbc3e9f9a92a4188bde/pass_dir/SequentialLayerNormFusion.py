import torch
import triton
import triton.language as tl
import math

def pattern(x, weight1, bias1, weight2, bias2):
    ln1 = torch.nn.functional.layer_norm(x, (768,), weight1, bias1, 1e-05)
    ln2 = torch.nn.functional.layer_norm(ln1, (768,), weight2, bias2, 1e-05)
    return ln1, ln2

def replacement_args(x, weight1, bias1, weight2, bias2):
    hidden_size = 768 if x.size(-1) == 768 else 1408
    return (x, weight1, bias1, weight2, bias2, hidden_size)

@triton.jit
def fused_layernorm_kernel(
    x_ptr,
    weight1_ptr, bias1_ptr,
    weight2_ptr, bias2_ptr,
    out1_ptr, out2_ptr,
    n_elements, hidden_size,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    offset = row_idx * hidden_size
    
    # Load input row
    x = tl.load(x_ptr + offset, mask=offset < n_elements).to(tl.float32)
    
    # LayerNorm 1
    mean = x.to(tl.float32)
    var = mean * mean
    mean_rms = tl.sqrt(mean * mean + var)
    weight1 = tl.load(weight1_ptr + tl.arange(0, hidden_size), mask=tl.arange(0, hidden_size) < hidden_size).to(tl.float32)
    bias1 = tl.load(bias1_ptr + tl.arange(0, hidden_size), mask=tl.arange(0, hidden_size) < hidden_size).to(tl.float32)
    
    out1 = (x - mean_rms) * (weight1 / tl.sqrt(var + eps)) + bias1
    
    # LayerNorm 2  
    mean2 = out1.to(tl.float32)
    var2 = mean2 * mean2
    mean_rms2 = tl.sqrt(mean2 * mean2 + var2)
    weight2 = tl.load(weight2_ptr + tl.arange(0, hidden_size), mask=tl.arange(0, hidden_size) < hidden_size).to(tl.float32)
    bias2 = tl.load(bias2_ptr + tl.arange(0, hidden_size), mask=tl.arange(0, hidden_size) < hidden_size).to(tl.float32)
    
    out2 = (out1 - mean_rms2) * (weight2 / tl.sqrt(var2 + eps)) + bias2
    
    # Store results
    tl.store(out1_ptr + offset, out1, mask=offset < n_elements)
    tl.store(out2_ptr + offset, out2, mask=offset < n_elements)

@torch.fx.wrap
def fused_layernorm_operation(x, weight1, bias1, weight2, bias2, hidden_size):
    n_elements = x.numel()
    
    # For large tensors, use better tiling
    if hidden_size >= 768:
        block_size = 256
    else:
        block_size = 128
        
    num_programs = (n_elements + block_size - 1) // block_size
    
    out1 = torch.empty_like(x)
    out2 = torch.empty_like(x)
    
    fused_layernorm_kernel[(num_programs,)](
        x_ptr=x,
        weight1_ptr=weight1,
        bias1_ptr=bias1,
        weight2_ptr=weight2,
        bias2_ptr=bias2,
        out1_ptr=out1,
        out2_ptr=out2,
        n_elements=n_elements,
        hidden_size=hidden_size,
        eps=1e-05,
        BLOCK_SIZE=block_size
    )
    
    return out1, out2

def replacement_func():
    return fused_layernorm_operation