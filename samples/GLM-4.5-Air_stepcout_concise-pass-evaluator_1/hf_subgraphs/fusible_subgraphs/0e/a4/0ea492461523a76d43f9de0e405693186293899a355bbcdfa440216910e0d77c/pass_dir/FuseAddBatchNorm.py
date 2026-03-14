import torch
import triton
import triton.language as tl

def pattern(arg4, arg5, arg0, arg1, arg2, arg3):
    tmp_0 = arg0
    tmp_1 = arg1
    tmp_2 = arg2
    tmp_3 = arg3
    arg5 += arg4
    tmp_4 = arg5
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
    return (tmp_4, tmp_5)

def replacement_args(in_4, in_5, in_0, in_1, in_2, in_3):
    return (in_4, in_5, in_0, in_1, in_2, in_3)

@triton.jit
def fused_add_batch_norm_kernel(x_ptr, y_ptr, gamma_ptr, beta_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr, 
                               added_out_ptr, bn_out_ptr, n_elements, BLOCK_SIZE: tl.constexpr, eps=1e-05, momentum=0.1):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    gamma = tl.load(gamma_ptr, mask=mask, other=1.0)
    beta = tl.load(beta_ptr, mask=mask, other=0.0)
    running_mean = tl.load(running_mean_ptr, mask=mask, other=0.0)
    running_var = tl.load(running_var_ptr, mask=mask, other=1.0)
    weight = tl.load(weight_ptr, mask=mask, other=0.0)
    bias = tl.load(bias_ptr, mask=mask, other=0.0)
    
    # Perform addition: x + y
    added = x + y
    
    # Perform batch normalization: (added * weight + bias - running_mean) / sqrt(running_var + eps) * gamma + beta
    normalized = ((added * weight + bias - running_mean) / tl.sqrt(running_var + eps)) * gamma + beta
    
    # Store results
    tl.store(added_out_ptr + offsets, added, mask=mask)
    tl.store(bn_out_ptr + offsets, normalized, mask=mask)

@torch.fx.wrap
def fused_add_batch_norm(x, y, gamma, beta, running_mean, running_var, weight, bias):
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensors
    added_out = torch.empty_like(x)
    bn_out = torch.empty_like(x)
    
    # Launch kernel
    fused_add_batch_norm_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        gamma_ptr=gamma,
        beta_ptr=beta,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        added_out_ptr=added_out,
        bn_out_ptr=bn_out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
        eps=1e-05,
        momentum=0.1,
    )
    
    return added_out, bn_out

def replacement_func():
    return fused_add_batch_norm