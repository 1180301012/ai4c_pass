import torch
import triton
import triton.language as tl

def pattern(input_tensor, weight, bias):
    # Layer norm on full tensor
    normalized = torch.nn.functional.layer_norm(input_tensor, (512,), weight, bias, 1e-06)
    # Slice only the first element along dimension 1
    result = normalized[slice(None, None, None), 0]
    return result

def replacement_args(input_tensor, weight, bias):
    return (input_tensor, weight, bias)

@triton.jit
def optimized_layer_norm_slice_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_batch: tl.constexpr,
    n_rows: tl.constexpr,
    n_features: tl.constexpr,
    eps: float,
):
    """Optimized kernel that computes layer norm only for the required slice"""
    pid = tl.program_id(0)
    
    # Each thread handles one batch element
    if pid >= n_batch:
        return
    
    batch_idx = pid
    target_row = 0  # We only need row 0 for each batch
    
    # Calculate base offset for this batch and target row
    row_offset = batch_idx * n_rows * n_features + target_row * n_features
    
    # Compute mean and variance for the target row (all features)
    row_sum = 0.0
    row_sum_sq = 0.0
    
    for i in range(n_features):
        val = tl.load(input_ptr + (row_offset + i), mask=i < n_features, other=0.0)
        row_sum += val
        row_sum_sq += val * val
    
    mean = row_sum / n_features
    var = (row_sum_sq / n_features) - (mean * mean)
    
    # Apply layer normalization to each feature in the target row
    for i in range(n_features):
        x = tl.load(input_ptr + (row_offset + i), mask=i < n_features, other=0.0)
        w = tl.load(weight_ptr + i, mask=i < n_features, other=0.0)
        b = tl.load(bias_ptr + i, mask=i < n_features, other=0.0)
        
        # Layer normalization
        x_norm = (x - mean) / tl.sqrt(var + eps)
        y = x_norm * w + b
        
        # Store result at [batch_idx, i]
        output_offset = batch_idx * n_features + i
        tl.store(output_ptr + output_offset, y, mask=i < n_features)

@torch.fx.wrap
def optimized_layer_norm_with_slice(input_tensor, weight, bias):
    n_batch, n_rows, n_features = input_tensor.shape
    
    # Output shape is [n_batch, n_features] since we take only row 0
    output_shape = (n_batch, n_features)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Triton kernel configuration - one thread per batch element
    grid = (n_batch,)
    
    optimized_layer_norm_slice_kernel[grid](
        input_ptr=input_tensor,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        n_batch=n_batch,
        n_rows=n_rows,
        n_features=n_features,
        eps=1e-06,
    )
    
    return output

def replacement_func():
    return optimized_layer_norm_with_slice