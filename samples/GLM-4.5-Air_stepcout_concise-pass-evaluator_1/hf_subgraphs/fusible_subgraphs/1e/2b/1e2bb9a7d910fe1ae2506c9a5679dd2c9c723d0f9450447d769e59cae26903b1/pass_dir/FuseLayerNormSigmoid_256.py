import torch
import triton
import triton.language as tl
import math

def pattern(in_3, in_1, in_0):
    """
    Match layer_norm followed by sigmoid fusion pattern.
    Original: tmp_2 = torch.nn.functional.layer_norm(in_3, (256,), in_1, in_0, 1e-05)
              tmp_4 = tmp_2.sigmoid()
    """
    tmp_2 = torch.nn.functional.layer_norm(in_3, (256,), in_1, in_0, 1e-05)
    tmp_4 = tmp_2.sigmoid()
    return tmp_4

def replacement_args(in_3, in_1, in_0):
    return (in_3, in_1, in_0)

@triton.jit
def sigmoid_fused_kernel(
    x_ptr,          # Input tensor after layer_norm [300, 1, 256]
    y_ptr,          # Output tensor [300, 1, 256]
    n_elements,     # Total elements in tensor
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data (already normalized by PyTorch layer_norm)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply sigmoid activation using exponential function
    # sigmoid(x) = 1 / (1 + exp(-x))
    neg_x = -x
    exp_neg_x = tl.exp(neg_x)
    out = 1.0 / (1.0 + exp_neg_x)
    
    # Store result
    tl.store(y_ptr + offsets, out, mask=mask)

@triton.jit
def compute_layer_norm_params_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    n_samples,          # 300
    norm_dim_size,      # 256
    n_elements,         # Total elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= n_samples:
        return
    
    # Each program handles one sample
    sample_start = pid * norm_dim_size
    offsets = sample_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (sample_start + norm_dim_size)
    
    # Load the sample data
    x_sample = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute mean and variance for this sample
    sample_mean = tl.sum(x_sample) / norm_dim_size
    sample_var = tl.sum((x_sample - sample_mean) * (x_sample - sample_mean)) / norm_dim_size
    
    # Store results at the sample index
    tl.store(mean_ptr + pid, sample_mean)
    tl.store(var_ptr + pid, sample_var)

@triton.jit
def apply_layer_norm_sigmoid_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    var_ptr,
    out_ptr,
    n_samples,          # 300
    norm_dim_size,      # 256
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (n_samples * norm_dim_size)
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute sample index and position within sample
    sample_id = offsets // norm_dim_size
    pos_in_sample = offsets % norm_dim_size
    
    # Load mean, variance, weight, and bias for this position
    sample_mean = tl.load(mean_ptr + sample_id, mask=sample_id < n_samples, other=0.0)
    sample_var = tl.load(var_ptr + sample_id, mask=sample_id < n_samples, other=1.0)
    weight = tl.load(weight_ptr + pos_in_sample, mask=pos_in_sample < norm_dim_size, other=1.0)
    bias = tl.load(bias_ptr + pos_in_sample, mask=pos_in_sample < norm_dim_size, other=0.0)
    
    # Apply layer normalization
    x_norm = (x - sample_mean) / tl.sqrt(sample_var + eps)
    x_norm = x_norm * weight + bias
    
    # Apply sigmoid
    # sigmoid(x) = 1 / (1 + exp(-x))
    neg_x = -x_norm
    exp_neg_x = tl.exp(neg_x)
    out = 1.0 / (1.0 + exp_neg_x)
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_layer_norm_sigmoid(in_3, in_1, in_0):
    n_samples = 300
    norm_dim_size = 256
    total_elements = n_samples * norm_dim_size
    
    # Phase 1: Compute layer norm parameters (mean, var) for each sample
    mean_vals = torch.empty(n_samples, dtype=torch.float32, device=in_3.device)
    var_vals = torch.empty(n_samples, dtype=torch.float32, device=in_3.device)
    
    # Compute means and variances
    num_programs = (n_samples + 1023) // 1024
    compute_layer_norm_params_kernel[(num_programs,)](
        in_3,
        mean_vals,
        var_vals,
        n_samples,
        norm_dim_size,
        total_elements,
        BLOCK_SIZE=1024,
    )
    
    # Phase 2: Apply layer normalization and sigmoid
    out = torch.empty_like(in_3)
    num_programs = (total_elements + 1023) // 1024
    apply_layer_norm_sigmoid_kernel[(num_programs,)](
        in_3,
        in_1,
        in_0,
        mean_vals,
        var_vals,
        out,
        n_samples,
        norm_dim_size,
        1e-05,
        BLOCK_SIZE=1024,
    )
    
    return out

def replacement_func():
    return fused_layer_norm_sigmoid