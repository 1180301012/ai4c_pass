import torch
import triton
import triton.language as tl

def pattern(tmp_4, tmp_0, tmp_1, tmp_3, tmp_2):
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
    tmp_6 = torch.nn.functional.relu(tmp_5, inplace=True)
    return (tmp_4, tmp_6)

def replacement_args(tmp_4, tmp_0, tmp_1, tmp_3, tmp_2):
    return (tmp_4, tmp_0, tmp_1, tmp_3, tmp_2)

@triton.jit
def fused_batch_norm_relu_kernel(x_ptr, gamma_ptr, beta_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr,
                                out_ptr, n_elements, BLOCK_SIZE: tl.constexpr, eps=1e-05, momentum=0.1):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    gamma = tl.load(gamma_ptr, mask=mask, other=1.0)
    beta = tl.load(beta_ptr, mask=mask, other=0.0)
    running_mean = tl.load(running_mean_ptr, mask=mask, other=0.0)
    running_var = tl.load(running_var_ptr, mask=mask, other=1.0)
    weight = tl.load(weight_ptr, mask=mask, other=0.0)
    bias = tl.load(bias_ptr, mask=mask, other=0.0)
    
    # Perform batch normalization: (x * weight + bias - running_mean) / sqrt(running_var + eps) * gamma + beta
    normalized = ((x * weight + bias - running_mean) / tl.sqrt(running_var + eps)) * gamma + beta
    
    # Perform ReLU activation (inplace equivalent)
    relu_out = tl.maximum(normalized, 0.0)
    
    # Store result
    tl.store(out_ptr + offsets, relu_out, mask=mask)

@torch.fx.wrap
def fused_batch_norm_relu(x, gamma, beta, running_mean, running_var, weight, bias):
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    relu_out = torch.empty_like(x)
    
    # Launch kernel
    fused_batch_norm_relu_kernel[(num_programs,)](
        x_ptr=x,
        gamma_ptr=gamma,
        beta_ptr=beta,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=relu_out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
        eps=1e-05,
        momentum=0.1,
    )
    
    return x, relu_out

def replacement_func():
    return fused_batch_norm_relu