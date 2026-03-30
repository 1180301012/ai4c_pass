import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_0 = 0.5 * in_0
    tmp_1 = torch.pow(in_0, 3.0)
    tmp_2 = 0.044715 * tmp_1
    tmp_3 = in_0 + tmp_2
    tmp_4 = 0.7978845608028654 * tmp_3
    tmp_5 = torch.tanh(tmp_4)
    tmp_6 = 1.0 + tmp_5
    tmp_7 = tmp_0 * tmp_6
    return (tmp_7,)

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def custom_activation_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Constants
    c1 = 0.5
    c3 = 0.044715
    c4 = 0.7978845608028654
    c5 = 1.0
    
    # Fused computation:
    # tmp_0 = 0.5 * x
    # tmp_1 = x^3 (optimized as x*x*x for performance)
    # tmp_2 = 0.044715 * tmp_1
    # tmp_3 = x + tmp_2
    # tmp_4 = 0.7978845608028654 * tmp_3
    # tmp_5 = tanh(tmp_4)
    # tmp_6 = 1.0 + tmp_5
    # tmp_7 = tmp_0 * tmp_6
    
    tmp_0 = c1 * x
    tmp_1 = x * x * x  # Optimized cubic operation
    tmp_2 = c3 * tmp_1
    tmp_3 = x + tmp_2
    tmp_4 = c4 * tmp_3
    
    # Implement tanh using exponentials (optimized): tanh(x) = (exp(x) - 1/exp(x)) / (exp(x) + 1/exp(x))
    exp_x = tl.exp(tmp_4)
    exp_inv_x = 1.0 / exp_x  # Instead of exp(-tmp_4), use 1/exp(x)
    tmp_5 = (exp_x - exp_inv_x) / (exp_x + exp_inv_x)
    
    tmp_6 = c5 + tmp_5
    tmp_7 = tmp_0 * tmp_6
    
    # Store output
    tl.store(out_ptr + offsets, tmp_7, mask=mask)

@torch.fx.wrap
def custom_activation(x):
    n_elements = x.numel()
    
    # Optimize block size based on tensor size for better GPU utilization
    if n_elements < 1000000:
        BLOCK_SIZE = 512
    elif n_elements < 10000000:
        BLOCK_SIZE = 2048  # Optimal for medium-sized tensors like ours (6M elements)
    else:
        BLOCK_SIZE = 4096
    
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    custom_activation_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return custom_activation