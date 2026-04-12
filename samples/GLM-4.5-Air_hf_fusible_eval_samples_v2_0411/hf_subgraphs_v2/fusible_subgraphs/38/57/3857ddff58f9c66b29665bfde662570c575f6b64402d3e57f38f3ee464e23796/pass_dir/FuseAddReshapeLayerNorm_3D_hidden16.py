import torch
import triton
import triton.language as tl

# Pattern matching function - must match the exact computation in model.py
def pattern(in_2, in_3, in_1, in_0):
    # Match the computation exactly: addition -> reshape -> layer_norm
    tmp_2 = in_2 + in_3
    tmp_3 = tmp_2.reshape(-1, 16)  # Use hardcoded shape from model
    tmp_4 = torch.nn.functional.layer_norm(tmp_3, (16,), in_1, in_0, 1e-05)
    return (tmp_3, tmp_4)

# Argument extraction function
def replacement_args(in_2, in_3, in_1, in_0):
    return (in_2, in_3, in_1, in_0)

# Triton kernel for simple addition
@triton.jit
def add_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements: tl.constexpr,
):
    idx = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    mask = idx < n_elements
    
    x = tl.load(x_ptr + idx, mask=mask, other=0.0)
    y = tl.load(y_ptr + idx, mask=mask, other=0.0)
    tl.store(out_ptr + idx, x + y, mask=mask)

@torch.fx.wrap
def triton_add(x, y):
    n_elements = x.numel()
    out = torch.empty_like(x)
    add_kernel[(n_elements + 1023) // 1024](
        x_ptr=x, y_ptr=y, out_ptr=out, n_elements=n_elements
    )
    return out

# Triton kernel for layer normalization
@triton.jit
def layernorm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    n_elements: tl.constexpr,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
):
    # Each program processes one hidden dimension element for all positions
    hid_idx = tl.program_id(0)
    
    mask = hid_idx < hidden_size
    
    # Load weight and bias
    weight = tl.load(weight_ptr + hid_idx, mask=mask)
    bias = tl.load(bias_ptr + hid_idx, mask=mask)
    
    # Compute mean and variance across all positions
    sum_val = 0.0
    sum_sq = 0.0
    for i in range(0, n_elements, hidden_size):
        pos_idx = i + hid_idx
        if pos_idx < n_elements:
            x = tl.load(x_ptr + pos_idx)
            sum_val += x
            sum_sq += x * x
    
    mean = sum_val / (n_elements // hidden_size)
    var = sum_sq / (n_elements // hidden_size) - mean * mean
    inv_std = tl.rsqrt(var + eps)
    
    # Apply normalization
    for i in range(0, n_elements, hidden_size):
        pos_idx = i + hid_idx
        if pos_idx < n_elements:
            x = tl.load(x_ptr + pos_idx)
            normalized = (x - mean) * inv_std
            tl.store(out_ptr + pos_idx, normalized * weight + bias, mask=True)

@torch.fx.wrap  
def triton_layernorm(x, weight, bias):
    n_elements, hidden_size = x.shape
    out = torch.empty_like(x)
    layernorm_kernel[(hidden_size,)](
        x_ptr=x, weight_ptr=weight, bias_ptr=bias, out_ptr=out,
        n_elements=n_elements, hidden_size=hidden_size, eps=1e-05
    )
    return out

# Fused function using Triton kernels
def fused_add_reshape_layernorm_triton(in_2, in_3, weight, bias):
    # Get input shapes
    batch_size, seq_len, hidden_size = in_2.shape
    
    # Step 1: Addition using Triton
    add_result = triton_add(in_2, in_3)
    
    # Step 2: Reshape
    reshape_result = add_result.reshape(-1, hidden_size)
    
    # Step 3: Layer norm using Triton
    norm_result = triton_layernorm(reshape_result, weight, bias)
    
    return reshape_result, norm_result

# Replacement function
def replacement_func():
    return fused_add_reshape_layernorm_triton