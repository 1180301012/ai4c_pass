import torch
import triton
import triton.language as tl

def pattern(emb_input, weight, bias):
    return torch.nn.functional.layer_norm(emb_input, (weight.numel(),), weight, bias, 1e-05)

def replacement_args(emb_input, weight, bias):
    return (emb_input, weight, bias)

@triton.jit
def naive_mean_kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    mask = offset < n_elements

    # Initialize local sum
    local_sum = tl.zeros(1, dtype=tl.float32)
    
    # Process chunk
    for i in range(0, BLOCK_SIZE):
        idx = offset + i
        if idx < n_elements:
            val = tl.load(x_ptr + idx)
            local_sum += val
    
    # Return local sum for reduction
    return local_sum if pid * BLOCK_SIZE < n_elements else 0.0

@triton.jit
def mean_kernel(x_ptr, mean_ptr, n_elements, elements_per_thread, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    
    # Load a chunk of data
    offsets = tl.arange(0, BLOCK_SIZE) + offset
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute partial sum
    partial_sum = tl.sum(x)
    
    # Store partial sum
    if pid * BLOCK_SIZE < n_elements:
        tl.store(mean_ptr + pid, partial_sum)

@triton.jit
def variance_kernel(x_ptr, mean_ptr, var_ptr, n_elements, elements_per_thread, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    
    # Load mean
    mean = tl.load(mean_ptr)
    
    # Load chunk and compute variance
    offsets = tl.arange(0, BLOCK_SIZE) + offset
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute partial variance
    diff = x - mean
    partial_var = tl.sum(diff * diff)
    
    # Store partial variance
    if pid * BLOCK_SIZE < n_elements:
        tl.store(var_ptr + pid, partial_var)

@triton.jit
def layernorm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    mean_val,
    var_val,
    eps,
    n_elements,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element in the batch
    pid = tl.program_id(0)
    offset = pid * hidden_size
    
    # Load weight and bias
    weight = tl.load(weight_ptr)
    bias = tl.load(bias_ptr)
    
    # Process hidden dimension
    offsets = tl.arange(0, BLOCK_SIZE) + offset
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Normalize: (x - mean) / sqrt(var + eps) * weight + bias
    normalized = (x - mean_val) / tl.sqrt(var_val + eps)
    result = normalized * weight + bias
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def optimized_layer_norm(x, weight, bias):
    device = x.device
    dtype = x.dtype
    batch_size, seq_len, hidden_size = x.shape
    n_elements = batch_size * seq_len * hidden_size
    
    # Convert to float32 for numerical stability
    x_float = x.to(torch.float32)
    
    # Compute mean
    mean = x_float.mean()
    
    # Compute variance
    var = ((x_float - mean) ** 2).mean()
    
    # Normalize and apply scale/bias
    eps = 1e-05
    var_sqrt = torch.sqrt(var + eps)
    x_normalized = (x_float - mean) / var_sqrt
    
    # Apply weight and bias
    # Handle weight/bias broadcasting
    if weight.dim() == 1:
        weight = weight.view(1, 1, -1)
    if bias.dim() == 1:
        bias = bias.view(1, 1, -1)
    
    result = x_normalized * weight + bias
    
    # Convert back to original dtype
    return result.to(dtype)

def replacement_func():
    return optimized_layer_norm