import torch
import triton
import triton.language as tl

def pattern(x, weight, bias):
    # Layer norm followed by sigmoid - exact match from model
    tmp_ln = torch.nn.functional.layer_norm(x, (256,), weight, bias, 1e-05)
    tmp_sigmoid = torch.sigmoid(tmp_ln)
    return tmp_ln, tmp_sigmoid

def replacement_args(x, weight, bias):
    return (x, weight, bias)

@triton.jit
def simple_layer_norm_sigmoid_kernel(
    x_ptr, weight_ptr, bias_ptr,
    ln_out_ptr, sigmoid_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    eps: tl.constexpr = 1e-05
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    
    # Compute mean and variance for layer norm
    block_mean = tl.sum(x) / n_elements
    x_centered = x - block_mean
    block_var = tl.sum(x_centered * x_centered) / n_elements
    block_std = tl.sqrt(block_var + eps)
    
    # Layer norm + sigmoid
    ln_out = (x - block_mean) / block_std * weight + bias
    sigmoid_out = 1.0 / (1.0 + tl.exp(-ln_out))
    
    # Store results
    tl.store(ln_out_ptr + offsets, ln_out, mask=mask)
    tl.store(sigmoid_out_ptr + offsets, sigmoid_out, mask=mask)

@torch.fx.wrap
def simple_layer_norm_sigmoid(x, weight, bias):
    # Get tensor size
    n_elements = x.numel()
    
    # Determine block size and grid size
    BLOCK_SIZE = 1024  # Optimal for GPU
    num_programs = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Allocate output tensors
    ln_out = torch.empty_like(x)
    sigmoid_out = torch.empty_like(x)
    
    # Launch kernel
    simple_layer_norm_sigmoid_kernel[(num_programs,)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        ln_out_ptr=ln_out,
        sigmoid_out_ptr=sigmoid_out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        eps=1e-05
    )
    
    return ln_out, sigmoid_out

def replacement_func():
    return simple_layer_norm_sigmoid