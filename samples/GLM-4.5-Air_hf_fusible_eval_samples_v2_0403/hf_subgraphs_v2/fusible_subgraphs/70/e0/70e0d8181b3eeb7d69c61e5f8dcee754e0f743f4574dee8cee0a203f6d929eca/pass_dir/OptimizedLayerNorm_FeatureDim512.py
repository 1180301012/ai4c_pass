import torch
import triton
import triton.language as tl

# Pattern matching for layer norm - matches exactly the computation in model.py
def layer_norm_pattern(x, weight, bias, normalized_shape, eps):
    """Match layer_norm with the exact signature from model.py"""
    return torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps)

def replacement_args(x, weight, bias, normalized_shape, eps):
    """Extract arguments needed for the optimized kernel"""
    return (x, weight, bias, eps)

@triton.jit
def optimized_layer_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    n_features, n_elements, eps,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    """Optimized LayerNorm kernel using Triton with block-based parallelization"""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(n_features, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(n_elements, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_in_group = group_id * num_pid_in_group
    last_pid_in_group = min(first_pid_in_group + num_pid_in_group, num_pid_m * num_pid_n)
    pid_in_group = pid - first_pid_in_group

    block_id_m = pid_in_group // num_pid_n
    block_id_n = pid_in_group % num_pid_n

    offsets_m = block_id_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = block_id_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offsets_m[:, None] < n_features) & (offsets_n[None, :] < n_elements)
    
    # Load input slice
    x = tl.load(x_ptr + offsets_m[:, None] * n_elements + offsets_n[None, :], mask=mask, other=0.0)
    
    # Compute mean
    current_mean = tl.sum(x, axis=1) / n_features
    
    # Compute variance (using Welford's algorithm for numerical stability)
    x_centered = x - current_mean[:, None]
    current_var = tl.sum(x_centered * x_centered, axis=1) / n_features
    
    # Normalize
    x_norm = (x - current_mean[:, None]) / tl.sqrt(current_var[:, None] + eps)
    
    # Load weights and bias
    if weight_ptr is not None:
        weights = tl.load(weight_ptr + offsets_m, mask=(offsets_m < n_features), other=1.0)
    else:
        weights = tl.full((BLOCK_SIZE_M,), 1.0, dtype=tl.float32)
        
    if bias_ptr is not None:
        biases = tl.load(bias_ptr + offsets_m, mask=(offsets_m < n_features), other=0.0)
    else:
        biases = tl.full((BLOCK_SIZE_M,), 0.0, dtype=tl.float32)
    
    # Apply weight and bias
    out = x_norm * weights[:, None] + biases[:, None]
    
    # Store output
    tl.store(out_ptr + offsets_m[:, None] * n_elements + offsets_n[None, :], out, mask=mask)

@torch.fx.wrap
def optimized_layer_norm(x, weight, bias, eps):
    """Wrapper for optimized layer norm kernel"""
    if x.dim() != 3:
        raise ValueError("Expected 3D input tensor")
    
    # Get tensor dimensions [batch, seq_len, features]
    batch_size, seq_len, n_features = x.shape
    n_elements = batch_size * seq_len * n_features
    
    # Determine block sizes based on feature dimension
    if n_features <= 512:
        BLOCK_SIZE_M = 32
        BLOCK_SIZE_N = 128
        GROUP_SIZE_M = 4
    else:
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 64
        GROUP_SIZE_M = 8
    
    # Calculate grid size
    num_pid_m = tl.cdiv(n_features, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(batch_size * seq_len, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    num_groups = (num_pid_m * num_pid_n + num_pid_in_group - 1) // num_pid_in_group
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch kernel
    optimized_layer_norm_kernel[(num_groups,)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        n_features=n_features,
        n_elements=n_elements,
        eps=eps,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        GROUP_SIZE_M=GROUP_SIZE_M
    )
    
    return out

def replacement_func():
    """Return the optimized layer norm function"""
    return optimized_layer_norm